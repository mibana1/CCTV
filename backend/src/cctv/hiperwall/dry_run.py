"""Translate rule events into persisted Hiperwall DRY RUN operations."""

from __future__ import annotations

import logging
from collections.abc import Collection

from cctv.core.execution import external_action_allowed
from cctv.core.settings import AppMode, Settings
from cctv.hiperwall.mapping import mapping_from_rule
from cctv.hiperwall.models import DisplayAction
from cctv.rules import RuleDefinition, RuleEvent

logger = logging.getLogger(__name__)


class HiperwallDryRunPlanner:
    """Build Hiperwall work records while guaranteeing that no request is sent."""

    def __init__(
        self,
        settings: Settings,
        rules: Collection[RuleDefinition] = (),
    ) -> None:
        if settings.app_mode is not AppMode.DRY_RUN:
            raise ValueError("HiperwallDryRunPlanner requires CCTV_APP_MODE=dry_run")
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
        """Map every emitted rule event to one deterministic simulated operation."""
        actions: list[DisplayAction] = []
        for event in events:
            action_type = "restore_layout" if event.event_state == "ended" else "open_source"
            action_name = f"hiperwall_{action_type}"
            if external_action_allowed(self.settings, action=action_name):
                raise RuntimeError("DRY RUN planner must never allow an external action")

            request = {
                "operation": action_type,
                "source_name": source_name,
                "analysis_run_id": analysis_run_id,
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
            mapping = self._mappings.get(event.rule_id)
            rule = self._rules.get(event.rule_id)
            if mapping is not None:
                request["target"] = mapping.target_request()
                request["display_seconds"] = mapping.display_seconds
                if event.event_state == "occurred":
                    request["close_after_seconds"] = mapping.display_seconds
            if rule is not None:
                request["label"] = f"{rule.name} | {source_name} | {event.event_type}"
            result = {
                "external_request_sent": False,
                "reason": "dry_run",
            }
            action = DisplayAction(
                rule_event_id=event.id,
                action_type=action_type,
                request=request,
                result=result,
            )
            actions.append(action)
            logger.info(
                "Hiperwall DRY RUN action planned",
                extra={
                    "event": "hiperwall_dry_run_action_planned",
                    "display_action_id": action.id,
                    "rule_event_id": event.id,
                    "rule_id": event.rule_id,
                    "analysis_run_id": analysis_run_id,
                    "source_name": source_name,
                    "track_id": event.track_id,
                    "rule_event_type": event.event_type,
                    "action_type": action_type,
                    "external_request_sent": False,
                },
            )
        return tuple(actions)


__all__ = ["HiperwallDryRunPlanner"]
