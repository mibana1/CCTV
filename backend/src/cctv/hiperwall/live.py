"""LIVE action planning and durable HiperInterface job execution."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from math import isfinite
from time import monotonic

from cctv.core.execution import external_action_allowed
from cctv.core.settings import AppMode, Settings
from cctv.db.display_actions import DisplayActionRecord, DisplayActionRepository
from cctv.hiperwall.client import HiperwallClient, HiperwallRequestError
from cctv.hiperwall.mapping import HiperwallMapping, mapping_from_rule
from cctv.hiperwall.models import DisplayAction
from cctv.hiperwall.reconciliation import HiperwallReconciler
from cctv.rules import RuleDefinition, RuleEvent

logger = logging.getLogger(__name__)


class HiperwallLivePlanner:
    """Translate mapped rule events into persistent LIVE queue records."""

    def __init__(self, settings: Settings, rules: Collection[RuleDefinition]) -> None:
        if settings.app_mode is not AppMode.LIVE:
            raise ValueError("HiperwallLivePlanner requires CCTV_APP_MODE=live")
        self.settings = settings
        self._rules = {rule.id: rule for rule in rules}
        self._mappings = {
            rule.id: mapping
            for rule in rules
            if (
                mapping := mapping_from_rule(
                    rule,
                    default_display_seconds=settings.hiperwall_default_display_seconds,
                )
            )
            is not None
        }

    def plan(
        self,
        events: Collection[RuleEvent],
        *,
        analysis_run_id: str,
        source_name: str,
    ) -> tuple[DisplayAction, ...]:
        if not external_action_allowed(self.settings, action="hiperwall_queue"):
            raise RuntimeError("LIVE Hiperwall planning requires external actions")
        planned: list[DisplayAction] = []
        now = datetime.now(UTC)
        for event in events:
            rule = self._rules.get(event.rule_id)
            mapping = self._mappings.get(event.rule_id)
            if rule is None or mapping is None:
                logger.debug(
                    "Hiperwall LIVE action skipped because the rule is not mapped",
                    extra={
                        "event": "hiperwall_live_mapping_missing",
                        "rule_event_id": event.id,
                        "rule_id": event.rule_id,
                        "source_name": source_name,
                    },
                )
                continue
            instance_id = _instance_id(event, source_name)
            request = _request(
                event,
                rule,
                mapping,
                analysis_run_id=analysis_run_id,
                source_name=source_name,
                instance_id=instance_id,
            )
            action_type = "restore_layout" if event.event_state == "ended" else "open_source"
            if event.event_state == "occurred":
                request["close_after_seconds"] = mapping.display_seconds
                request["display_cooldown_seconds"] = _display_cooldown_seconds(rule)
            planned.append(
                DisplayAction(
                    rule_event_id=event.id,
                    action_type=action_type,
                    request={**request, "operation": action_type},
                    result={"external_request_sent": False, "reason": "queued"},
                    mode="live",
                    status="pending",
                    available_at=now,
                )
            )
        return tuple(planned)


class HiperwallActionWorker:
    """Poll durable LIVE actions and execute them outside the analysis process."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: HiperwallClient | None = None,
    ) -> None:
        if settings.app_mode is not AppMode.LIVE:
            raise ValueError("HiperwallActionWorker requires CCTV_APP_MODE=live")
        self.settings = settings
        self.repository = DisplayActionRepository(settings.database_path)
        token = (
            settings.hiperwall_token.get_secret_value()
            if settings.hiperwall_token is not None
            else ""
        )
        self.client = client or HiperwallClient(
            base_url=settings.hiperwall_base_url or "",
            auth_mode=settings.hiperwall_auth_mode,
            user=settings.hiperwall_user,
            token=token,
            timeout_seconds=settings.hiperwall_timeout_seconds,
        )
        self.reconciler = (
            HiperwallReconciler(settings, client=self.client)
            if settings.hiperwall_reconciliation_enabled
            else None
        )
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        recovered = self.repository.recover_interrupted(
            hold_for_reconciliation=self.reconciler is not None
        )
        self._thread = threading.Thread(
            target=self._run,
            name="hiperwall-action-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Hiperwall action worker started",
            extra={
                "event": "hiperwall_action_worker_started",
                "recovered_action_count": recovered,
                "reconciliation_enabled": self.reconciler is not None,
                "reconciliation_interval_seconds": (
                    self.settings.hiperwall_reconciliation_interval_seconds
                    if self.reconciler is not None
                    else None
                ),
                "reconciliation_force_close_enabled": (
                    self.settings.hiperwall_reconciliation_force_close_enabled
                    if self.reconciler is not None
                    else None
                ),
            },
        )

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.settings.hiperwall_timeout_seconds + 1))
        self.client.close()
        logger.info(
            "Hiperwall action worker stopped",
            extra={"event": "hiperwall_action_worker_stopped"},
        )

    def process_once(self) -> bool:
        action = self.repository.claim_next(
            allow_uncertain_outcomes=self.reconciler is None
        )
        if action is None:
            return False
        try:
            result = self.client.execute(action.request)
        except HiperwallRequestError as error:
            self._record_failure(action.id, action.attempt_count, error)
        except Exception as error:  # pragma: no cover - final safety boundary
            logger.exception(
                "Unexpected Hiperwall worker failure",
                extra={"event": "hiperwall_action_unexpected_failure", "action_id": action.id},
            )
            self._record_failure(
                action.id,
                action.attempt_count,
                HiperwallRequestError(
                    "hiperwall_unexpected_error",
                    f"Unexpected Hiperwall error: {type(error).__name__}",
                ),
            )
        else:
            self.repository.mark_succeeded(
                action.id,
                result=result,
                followup_action=_scheduled_close(action),
            )
            logger.info(
                "Hiperwall action completed",
                extra={
                    "event": "hiperwall_action_completed",
                    "action_id": action.id,
                    "rule_event_id": action.rule_event_id,
                    "action_type": action.action_type,
                    "attempt_count": action.attempt_count,
                },
            )
        return True

    def _record_failure(
        self,
        action_id: str,
        attempt_count: int,
        error: HiperwallRequestError,
    ) -> None:
        should_retry = error.retryable and attempt_count < self.settings.hiperwall_max_attempts
        delay_seconds = min(
            self.settings.hiperwall_retry_base_seconds * (2 ** max(attempt_count - 1, 0)),
            self.settings.hiperwall_retry_max_seconds,
        )
        status = self.repository.mark_failed(
            action_id,
            error_code=error.code,
            error_message=str(error),
            retry_after_seconds=delay_seconds if should_retry else None,
        )
        logger.warning(
            "Hiperwall action failed",
            extra={
                "event": "hiperwall_action_failed",
                "action_id": action_id,
                "error_code": error.code,
                "error_message": str(error),
                "error_retryable": error.retryable,
                "attempt_count": attempt_count,
                "action_status": status,
            },
        )

    def _run(self) -> None:
        next_reconciliation_at = monotonic()
        while not self._stop.is_set():
            if self.reconciler is not None and monotonic() >= next_reconciliation_at:
                next_reconciliation_at = (
                    monotonic() + self.settings.hiperwall_reconciliation_interval_seconds
                )
                try:
                    self.reconciler.run_once()
                except Exception:
                    logger.exception(
                        "Hiperwall action worker reconciliation failed",
                        extra={"event": "hiperwall_action_worker_reconciliation_failed"},
                    )
            try:
                processed = 0
                while not self._stop.is_set() and processed < 100 and self.process_once():
                    processed += 1
            except Exception:
                logger.exception(
                    "Hiperwall action worker polling failed",
                    extra={"event": "hiperwall_action_worker_poll_failed"},
                )
                try:
                    self.repository.recover_interrupted(
                        hold_for_reconciliation=self.reconciler is not None
                    )
                except Exception:
                    logger.exception(
                        "Hiperwall action recovery failed",
                        extra={"event": "hiperwall_action_recovery_failed"},
                    )
            self._wake.wait(self.settings.hiperwall_poll_interval_seconds)
            self._wake.clear()


def _request(
    event: RuleEvent,
    rule: RuleDefinition,
    mapping: HiperwallMapping,
    *,
    analysis_run_id: str,
    source_name: str,
    instance_id: str,
) -> dict[str, object]:
    return {
        "source_name": source_name,
        "analysis_run_id": analysis_run_id,
        "instance_id": instance_id,
        "target": mapping.target_request(),
        "label": f"{rule.name} | {source_name} | {event.event_type}",
        "rule_event": {
            "id": event.id,
            "rule_id": event.rule_id,
            "track_id": event.track_id,
            "event_type": event.event_type,
            "event_state": event.event_state,
            "occurred_at_seconds": event.occurred_at_seconds,
            "class_name": event.class_name,
            "confidence": event.confidence,
            "payload": event.payload,
        },
    }


def _instance_id(event: RuleEvent, source_name: str) -> str:
    lifecycle_key = event.id if event.event_state == "occurred" else str(event.track_id)
    source = re.sub(r"[^A-Za-z0-9_.-]", "-", source_name).strip("-")[:36] or "source"
    digest = hashlib.sha256(f"{source_name}|{event.rule_id}|{lifecycle_key}".encode()).hexdigest()[
        :20
    ]
    return f"cctv-{source}-{digest}"


def _display_cooldown_seconds(rule: RuleDefinition) -> float:
    value = rule.parameters.get("cooldown_seconds", 0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0 <= float(value) <= 3_600
    ):
        raise ValueError("cooldown_seconds must be between 0 and 3600")
    return float(value)


def _scheduled_close(action: DisplayActionRecord) -> DisplayAction | None:
    if action.action_type != "open_source":
        return None
    delay = action.request.get("close_after_seconds")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        return None
    close_request = dict(action.request)
    close_request.pop("close_after_seconds", None)
    close_request["operation"] = "restore_layout"
    return DisplayAction(
        rule_event_id=action.rule_event_id,
        action_type="restore_layout",
        request=close_request,
        result={"external_request_sent": False, "reason": "scheduled"},
        mode="live",
        status="pending",
        available_at=datetime.now(UTC) + timedelta(seconds=delay),
    )


__all__ = ["HiperwallActionWorker", "HiperwallLivePlanner"]
