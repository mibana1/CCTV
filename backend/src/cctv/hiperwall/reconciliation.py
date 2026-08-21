"""Reconcile durable Hiperwall actions with currently open external instances."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from cctv.core.settings import AppMode, Settings
from cctv.db.display_actions import (
    INTERRUPTED_UNKNOWN_OUTCOME,
    DisplayActionRecord,
    DisplayActionRepository,
)
from cctv.db.display_states import DisplayStateRecord, DisplayStateRepository
from cctv.db.maintenance_leases import MaintenanceLeaseRepository
from cctv.hiperwall.client import HiperwallClient, HiperwallRequestError
from cctv.hiperwall.models import DisplayAction

logger = logging.getLogger(__name__)

MANAGED_INSTANCE_PREFIX = "cctv-"
_LEASE_NAME = "hiperwall_reconciliation"
NowFactory = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class HiperwallReconciliationResult:
    """Summary of one lease-protected inventory comparison."""

    lease_acquired: bool
    inventory_instance_count: int = 0
    managed_external_instance_count: int = 0
    database_instance_count: int = 0
    mismatch_count: int = 0
    adopted_open_count: int = 0
    reconciled_close_count: int = 0
    forced_close_attempt_count: int = 0
    forced_close_succeeded_count: int = 0
    forced_close_failed_count: int = 0
    warning_only_count: int = 0


class HiperwallReconciler:
    """Compare app-owned instances with durable jobs and repair uncertain outcomes."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: HiperwallClient,
        action_repository: DisplayActionRepository | None = None,
        state_repository: DisplayStateRepository | None = None,
        lease_repository: MaintenanceLeaseRepository | None = None,
        owner_id: str | None = None,
        now_factory: NowFactory | None = None,
    ) -> None:
        if settings.app_mode is not AppMode.LIVE:
            raise ValueError("HiperwallReconciler requires CCTV_APP_MODE=live")
        self.settings = settings
        self.client = client
        self.action_repository = action_repository or DisplayActionRepository(
            settings.database_path
        )
        self.state_repository = state_repository or DisplayStateRepository(
            settings.database_path
        )
        self.lease_repository = lease_repository or MaintenanceLeaseRepository(
            settings.database_path
        )
        self.owner_id = owner_id or str(uuid4())
        self._now = now_factory or (lambda: datetime.now(UTC))

    def run_once(self) -> HiperwallReconciliationResult:
        """Read inventory once, repair mismatches, and never touch foreign instance IDs."""
        reconciled_at = _utc(self._now())
        acquired = self.lease_repository.acquire(
            _LEASE_NAME,
            self.owner_id,
            lease_seconds=self.settings.hiperwall_reconciliation_lease_seconds,
            now=reconciled_at,
        )
        if not acquired:
            result = HiperwallReconciliationResult(lease_acquired=False)
            logger.debug(
                "Hiperwall reconciliation skipped because another Backend owns the lease",
                extra={"event": "hiperwall_reconciliation_lease_unavailable"},
            )
            return result

        try:
            result = self._reconcile(reconciled_at=reconciled_at)
        except Exception as error:
            logger.exception(
                "Hiperwall reconciliation failed",
                extra={
                    "event": "hiperwall_reconciliation_failed",
                    "error_type": type(error).__name__,
                },
            )
            raise
        finally:
            try:
                self.lease_repository.release(_LEASE_NAME, self.owner_id)
            except Exception:
                logger.exception(
                    "Hiperwall reconciliation lease release failed",
                    extra={"event": "hiperwall_reconciliation_lease_release_failed"},
                )

        logger.info(
            "Hiperwall reconciliation completed",
            extra={"event": "hiperwall_reconciliation_completed", **asdict(result)},
        )
        return result

    def _reconcile(self, *, reconciled_at: datetime) -> HiperwallReconciliationResult:
        inventory = self.client.inventory()
        all_external_instances = _inventory_instance_ids(inventory)
        managed_external_instances = {
            instance_id
            for instance_id in all_external_instances
            if instance_id.startswith(MANAGED_INSTANCE_PREFIX)
        }
        actions = self.action_repository.list_for_reconciliation(
            instance_prefix=MANAGED_INSTANCE_PREFIX
        )
        states = self.state_repository.list_managed(now=reconciled_at)
        actions_by_instance = _actions_by_instance(actions)
        states_by_instance = {
            state.instance_id: state for state in states if state.instance_id is not None
        }

        mismatch_count = 0
        adopted_open_count = 0
        reconciled_close_count = 0
        forced_close_attempt_count = 0
        forced_close_succeeded_count = 0
        forced_close_failed_count = 0
        warning_only_count = 0

        database_instances = set(actions_by_instance) | set(states_by_instance)
        instance_ids = sorted(database_instances | managed_external_instances)
        for instance_id in instance_ids:
            instance_actions = actions_by_instance.get(instance_id, ())
            latest = instance_actions[-1] if instance_actions else None
            state = states_by_instance.get(instance_id)
            externally_open = instance_id in managed_external_instances

            if latest is None:
                if not externally_open:
                    continue
                mismatch_count += 1
                logger.warning(
                    "Managed Hiperwall instance has no durable action",
                    extra={
                        "event": "hiperwall_reconciliation_orphan_instance",
                        "instance_id": instance_id,
                    },
                )
                outcome = self._force_close(
                    instance_id,
                    reason="orphan_external_instance",
                    attempted_count=forced_close_attempt_count,
                )
                if outcome is None:
                    warning_only_count += 1
                else:
                    forced_close_attempt_count += 1
                    if outcome:
                        forced_close_succeeded_count += 1
                    else:
                        forced_close_failed_count += 1
                continue

            if externally_open and latest.action_type == "open_source":
                followup = _reconciled_followup(latest, reconciled_at=reconciled_at)
                needs_adoption = latest.status != "succeeded" or followup is not None
                if needs_adoption:
                    mismatch_count += 1
                    self.action_repository.reconcile_succeeded(
                        latest.id,
                        result=_reconciliation_result(
                            latest,
                            external_request_sent=True,
                            reason="external_open_observed",
                        ),
                        followup_action=followup,
                        completed_at=reconciled_at,
                    )
                    adopted_open_count += 1
                    logger.warning(
                        "Observed Hiperwall open was adopted into the durable queue",
                        extra={
                            "event": "hiperwall_reconciliation_open_adopted",
                            "instance_id": instance_id,
                            "action_id": latest.id,
                            "previous_status": latest.status,
                            "scheduled_close": followup is not None,
                        },
                    )
                continue

            if externally_open and latest.action_type == "restore_layout":
                if not _close_requires_force(
                    latest,
                    reconciled_at=reconciled_at,
                    grace_seconds=self.settings.hiperwall_reconciliation_grace_seconds,
                ):
                    continue
                mismatch_count += 1
                logger.warning(
                    "Hiperwall close is overdue or recorded complete but remains externally open",
                    extra={
                        "event": "hiperwall_reconciliation_force_close_required",
                        "instance_id": instance_id,
                        "action_id": latest.id,
                        "action_status": latest.status,
                        "last_error_code": latest.last_error_code,
                    },
                )
                outcome = self._force_close(
                    instance_id,
                    reason="close_not_reflected_externally",
                    attempted_count=forced_close_attempt_count,
                )
                if outcome is None:
                    warning_only_count += 1
                else:
                    forced_close_attempt_count += 1
                    if outcome:
                        self.action_repository.reconcile_succeeded(
                            latest.id,
                            result=_reconciliation_result(
                                latest,
                                external_request_sent=True,
                                reason="forced_close_succeeded",
                            ),
                            completed_at=reconciled_at,
                        )
                        forced_close_succeeded_count += 1
                        reconciled_close_count += 1
                    else:
                        forced_close_failed_count += 1
                continue

            if not externally_open and latest.action_type == "restore_layout":
                if latest.status == "succeeded" or _recent_action(
                    latest,
                    reconciled_at=reconciled_at,
                    grace_seconds=self.settings.hiperwall_reconciliation_grace_seconds,
                ):
                    continue
                mismatch_count += 1
                self.action_repository.reconcile_succeeded(
                    latest.id,
                    result=_reconciliation_result(
                        latest,
                        external_request_sent=False,
                        reason="external_instance_already_absent",
                    ),
                    completed_at=reconciled_at,
                )
                reconciled_close_count += 1
                logger.warning(
                    "Hiperwall close outcome was reconciled from inventory",
                    extra={
                        "event": "hiperwall_reconciliation_close_observed",
                        "instance_id": instance_id,
                        "action_id": latest.id,
                        "previous_status": latest.status,
                    },
                )
                continue

            if not externally_open and latest.action_type == "open_source":
                if _missing_open_requires_release(
                    latest,
                    state=state,
                    reconciled_at=reconciled_at,
                    grace_seconds=self.settings.hiperwall_reconciliation_grace_seconds,
                ):
                    mismatch_count += 1
                    self.action_repository.reconcile_missing_open(
                        latest.id,
                        reconciled_at=reconciled_at,
                    )
                    logger.warning(
                        "Uncertain Hiperwall open was released because no instance exists",
                        extra={
                            "event": "hiperwall_reconciliation_missing_open_released",
                            "instance_id": instance_id,
                            "action_id": latest.id,
                            "previous_status": latest.status,
                        },
                    )
                elif latest.status == "succeeded":
                    mismatch_count += 1
                    warning_only_count += 1
                    logger.warning(
                        "Database expects a Hiperwall instance that inventory does not contain",
                        extra={
                            "event": "hiperwall_reconciliation_expected_instance_missing",
                            "instance_id": instance_id,
                            "action_id": latest.id,
                        },
                    )

        return HiperwallReconciliationResult(
            lease_acquired=True,
            inventory_instance_count=len(all_external_instances),
            managed_external_instance_count=len(managed_external_instances),
            database_instance_count=len(database_instances),
            mismatch_count=mismatch_count,
            adopted_open_count=adopted_open_count,
            reconciled_close_count=reconciled_close_count,
            forced_close_attempt_count=forced_close_attempt_count,
            forced_close_succeeded_count=forced_close_succeeded_count,
            forced_close_failed_count=forced_close_failed_count,
            warning_only_count=warning_only_count,
        )

    def _force_close(
        self,
        instance_id: str,
        *,
        reason: str,
        attempted_count: int,
    ) -> bool | None:
        if (
            not self.settings.hiperwall_reconciliation_force_close_enabled
            or attempted_count
            >= self.settings.hiperwall_reconciliation_max_force_closes_per_run
        ):
            logger.warning(
                "Hiperwall mismatch requires manual close",
                extra={
                    "event": "hiperwall_reconciliation_force_close_deferred",
                    "instance_id": instance_id,
                    "reason": reason,
                    "force_close_enabled": (
                        self.settings.hiperwall_reconciliation_force_close_enabled
                    ),
                },
            )
            return None
        try:
            self.client.execute(
                {"operation": "restore_layout", "instance_id": instance_id}
            )
        except HiperwallRequestError as error:
            logger.error(
                "Hiperwall reconciliation force close failed",
                extra={
                    "event": "hiperwall_reconciliation_force_close_failed",
                    "instance_id": instance_id,
                    "reason": reason,
                    "error_code": error.code,
                    "error_message": str(error),
                    "error_retryable": error.retryable,
                },
            )
            return False
        except Exception as error:  # pragma: no cover - final adapter safety boundary
            logger.exception(
                "Unexpected Hiperwall reconciliation force close failure",
                extra={
                    "event": "hiperwall_reconciliation_force_close_failed",
                    "instance_id": instance_id,
                    "reason": reason,
                    "error_type": type(error).__name__,
                },
            )
            return False
        logger.warning(
            "Hiperwall reconciliation force close succeeded",
            extra={
                "event": "hiperwall_reconciliation_force_close_succeeded",
                "instance_id": instance_id,
                "reason": reason,
            },
        )
        return True


def _inventory_instance_ids(inventory: dict[str, object]) -> set[str]:
    contents = inventory.get("contents")
    if not isinstance(contents, list):
        return set()
    instance_ids: set[str] = set()
    for content in contents:
        if not isinstance(content, dict):
            continue
        instances = content.get("instances")
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_id = instance.get("id")
            if isinstance(instance_id, str) and instance_id.strip():
                instance_ids.add(instance_id.strip())
    return instance_ids


def _actions_by_instance(
    actions: tuple[DisplayActionRecord, ...],
) -> dict[str, tuple[DisplayActionRecord, ...]]:
    grouped: dict[str, list[DisplayActionRecord]] = defaultdict(list)
    for action in actions:
        instance_id = action.request.get("instance_id")
        if isinstance(instance_id, str) and instance_id.startswith(MANAGED_INSTANCE_PREFIX):
            grouped[instance_id].append(action)
    return {instance_id: tuple(items) for instance_id, items in grouped.items()}


def _reconciled_followup(
    action: DisplayActionRecord,
    *,
    reconciled_at: datetime,
) -> DisplayAction | None:
    delay = action.request.get("close_after_seconds")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        return None
    close_request = dict(action.request)
    close_request.pop("close_after_seconds", None)
    close_request["operation"] = "restore_layout"
    observed_open_at = action.last_attempt_at or action.updated_at
    available_at = max(reconciled_at, observed_open_at + timedelta(seconds=delay))
    return DisplayAction(
        rule_event_id=action.rule_event_id,
        action_type="restore_layout",
        request=close_request,
        result={
            "external_request_sent": False,
            "reason": "scheduled_by_reconciliation",
        },
        mode="live",
        status="pending",
        available_at=available_at,
    )


def _close_requires_force(
    action: DisplayActionRecord,
    *,
    reconciled_at: datetime,
    grace_seconds: float,
) -> bool:
    if action.last_error_code == INTERRUPTED_UNKNOWN_OUTCOME:
        return True
    if action.status in {"succeeded", "failed"}:
        return not _recent_action(
            action,
            reconciled_at=reconciled_at,
            grace_seconds=grace_seconds,
        )
    if action.status in {"pending", "retry"}:
        return reconciled_at >= action.available_at + timedelta(seconds=grace_seconds)
    if action.status == "processing":
        return not _recent_action(
            action,
            reconciled_at=reconciled_at,
            grace_seconds=grace_seconds,
        )
    return False


def _missing_open_requires_release(
    action: DisplayActionRecord,
    *,
    state: DisplayStateRecord | None,
    reconciled_at: datetime,
    grace_seconds: float,
) -> bool:
    if action.last_error_code == INTERRUPTED_UNKNOWN_OUTCOME:
        return True
    if state is None or state.state != "DISPLAYING":
        return False
    if action.status == "failed":
        return not _recent_action(
            action,
            reconciled_at=reconciled_at,
            grace_seconds=grace_seconds,
        )
    if action.status == "processing":
        return not _recent_action(
            action,
            reconciled_at=reconciled_at,
            grace_seconds=grace_seconds,
        )
    return False


def _recent_action(
    action: DisplayActionRecord,
    *,
    reconciled_at: datetime,
    grace_seconds: float,
) -> bool:
    reference = action.last_attempt_at or action.completed_at or action.updated_at
    return reconciled_at < reference + timedelta(seconds=grace_seconds)


def _reconciliation_result(
    action: DisplayActionRecord,
    *,
    external_request_sent: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "external_request_sent": external_request_sent,
        "operation": action.action_type,
        "instance_id": action.request.get("instance_id"),
        "reconciled": True,
        "reconciliation_reason": reason,
        "previous_status": action.status,
        "previous_error_code": action.last_error_code,
    }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("reconciliation timestamps must include timezone information")
    return value.astimezone(UTC)


__all__ = [
    "MANAGED_INSTANCE_PREFIX",
    "HiperwallReconciler",
    "HiperwallReconciliationResult",
]
