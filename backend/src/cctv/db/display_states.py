"""Durable rule/source lifecycle gate for LIVE Hiperwall displays."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Collection
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from cctv.db.database import connect_database

if TYPE_CHECKING:
    from cctv.hiperwall import DisplayAction
    from cctv.rules import RuleEvent


@dataclass(frozen=True, slots=True)
class DisplayStateRecord:
    """Current persisted display lifecycle for one rule and camera source."""

    rule_id: str
    source_name: str
    state: str
    active_rule_event_id: str | None
    open_action_id: str | None
    close_action_id: str | None
    instance_id: str | None
    cooldown_seconds: float
    displaying_since: datetime | None
    opened_at: datetime | None
    close_succeeded_at: datetime | None
    cooldown_until: datetime | None
    last_rule_event_id: str | None
    last_open_action_id: str | None
    last_close_action_id: str | None
    last_closed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class DisplayGateResult:
    """Events/actions accepted or suppressed by the durable display gate."""

    rule_events: tuple[RuleEvent, ...]
    display_actions: tuple[DisplayAction, ...]
    suppressed_rule_event_ids: tuple[str, ...]
    suppressed_display_action_ids: tuple[str, ...]


class DisplayStateRepository:
    """Inspect and advance persisted Hiperwall display lifecycle state."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def get(
        self,
        rule_id: str,
        source_name: str,
        *,
        now: datetime | None = None,
    ) -> DisplayStateRecord | None:
        with closing(connect_database(self.database_path)) as connection, connection:
            release_expired_display_states(connection, now=now)
            row = connection.execute(
                """
                SELECT * FROM hiperwall_display_states
                WHERE rule_id = ? AND source_name = ?
                """,
                (rule_id, source_name),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def release_expired_cooldowns(self, *, now: datetime | None = None) -> int:
        with closing(connect_database(self.database_path)) as connection, connection:
            return release_expired_display_states(connection, now=now)


def gate_live_display_events(
    connection: sqlite3.Connection,
    *,
    source_name: str,
    events: Collection[RuleEvent],
    actions: Collection[DisplayAction],
    now: datetime | None = None,
) -> DisplayGateResult:
    """Atomically admit at most one LIVE point event per rule/source lifecycle."""
    normalized_now = _utc_datetime(now)
    now_text = _utc_text(normalized_now)
    release_expired_display_states(connection, now=normalized_now)

    event_items = tuple(events)
    action_items = tuple(actions)
    actions_by_event: dict[str, list[DisplayAction]] = defaultdict(list)
    for action in action_items:
        actions_by_event[action.rule_event_id].append(action)

    accepted_events: list[RuleEvent] = []
    suppressed_event_ids: list[str] = []
    suppressed_action_ids: list[str] = []
    for event in event_items:
        event_actions = actions_by_event.get(event.id, [])
        gated_actions = [
            action for action in event_actions if _is_gated_live_action(event=event, action=action)
        ]
        if len(gated_actions) > 1:
            raise ValueError("one rule event cannot create multiple gated LIVE open actions")
        if not gated_actions:
            accepted_events.append(event)
            continue
        action = gated_actions[0]
        if _claim_display_state(
            connection,
            event=event,
            action=action,
            source_name=source_name,
            now_text=now_text,
        ):
            accepted_events.append(event)
            continue
        suppressed_event_ids.append(event.id)
        suppressed_action_ids.extend(item.id for item in event_actions)

    suppressed = set(suppressed_event_ids)
    accepted_actions = tuple(
        action for action in action_items if action.rule_event_id not in suppressed
    )
    return DisplayGateResult(
        rule_events=tuple(accepted_events),
        display_actions=accepted_actions,
        suppressed_rule_event_ids=tuple(suppressed_event_ids),
        suppressed_display_action_ids=tuple(suppressed_action_ids),
    )


def record_display_action_success(
    connection: sqlite3.Connection,
    *,
    action_id: str,
    followup_action_id: str | None,
    completed_at: datetime,
) -> None:
    """Advance DISPLAYING only when the corresponding external action succeeded."""
    completed_text = _utc_text(completed_at)
    if followup_action_id is not None:
        connection.execute(
            """
            UPDATE hiperwall_display_states
            SET opened_at = COALESCE(opened_at, ?),
                close_action_id = ?,
                updated_at = ?
            WHERE open_action_id = ? AND state = 'DISPLAYING'
            """,
            (completed_text, followup_action_id, completed_text, action_id),
        )
        return

    row = connection.execute(
        """
        SELECT rule_id, source_name, cooldown_seconds
        FROM hiperwall_display_states
        WHERE close_action_id = ? AND state = 'DISPLAYING'
        """,
        (action_id,),
    ).fetchone()
    if row is None:
        return
    cooldown_until = completed_at + timedelta(seconds=float(row["cooldown_seconds"]))
    connection.execute(
        """
        UPDATE hiperwall_display_states
        SET state = 'COOLDOWN',
            close_succeeded_at = ?,
            cooldown_until = ?,
            last_rule_event_id = active_rule_event_id,
            last_open_action_id = open_action_id,
            last_close_action_id = close_action_id,
            last_closed_at = ?,
            updated_at = ?
        WHERE rule_id = ? AND source_name = ?
            AND close_action_id = ? AND state = 'DISPLAYING'
        """,
        (
            completed_text,
            _utc_text(cooldown_until),
            completed_text,
            completed_text,
            str(row["rule_id"]),
            str(row["source_name"]),
            action_id,
        ),
    )


_SAFE_OPEN_FAILURES = {
    "hiperwall_auth_failed",
    "hiperwall_command_forbidden",
    "hiperwall_content_not_found",
    "hiperwall_zone_not_found",
}


def release_failed_open_state(
    connection: sqlite3.Connection,
    *,
    action_id: str,
    error_code: str,
    failed_at: datetime,
) -> None:
    """Release only failures that prove the external open did not take effect."""
    if error_code not in _SAFE_OPEN_FAILURES:
        return
    failed_text = _utc_text(failed_at)
    connection.execute(
        """
        UPDATE hiperwall_display_states
        SET state = 'IDLE',
            last_rule_event_id = active_rule_event_id,
            last_open_action_id = open_action_id,
            active_rule_event_id = NULL,
            open_action_id = NULL,
            close_action_id = NULL,
            instance_id = NULL,
            cooldown_seconds = 0,
            displaying_since = NULL,
            opened_at = NULL,
            close_succeeded_at = NULL,
            cooldown_until = NULL,
            updated_at = ?
        WHERE open_action_id = ? AND state = 'DISPLAYING'
        """,
        (failed_text, action_id),
    )


def release_expired_display_states(
    connection: sqlite3.Connection,
    *,
    now: datetime | None = None,
) -> int:
    """Move elapsed cooldown rows to IDLE while retaining last-session audit fields."""
    now_text = _utc_text(now)
    cursor = connection.execute(
        """
        UPDATE hiperwall_display_states
        SET state = 'IDLE',
            active_rule_event_id = NULL,
            open_action_id = NULL,
            close_action_id = NULL,
            instance_id = NULL,
            cooldown_seconds = 0,
            displaying_since = NULL,
            opened_at = NULL,
            close_succeeded_at = NULL,
            cooldown_until = NULL,
            updated_at = ?
        WHERE state = 'COOLDOWN' AND cooldown_until <= ?
        """,
        (now_text, now_text),
    )
    return cursor.rowcount


def _is_gated_live_action(*, event: RuleEvent, action: DisplayAction) -> bool:
    return (
        event.event_state == "occurred"
        and action.mode == "live"
        and action.action_type == "open_source"
        and "close_after_seconds" in action.request
        and "display_cooldown_seconds" in action.request
    )


def _claim_display_state(
    connection: sqlite3.Connection,
    *,
    event: RuleEvent,
    action: DisplayAction,
    source_name: str,
    now_text: str,
) -> bool:
    instance_id = action.request.get("instance_id")
    cooldown_seconds = action.request.get("display_cooldown_seconds")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("gated LIVE action requires an instance_id")
    if (
        isinstance(cooldown_seconds, bool)
        or not isinstance(cooldown_seconds, (int, float))
        or not 0 <= float(cooldown_seconds) <= 3_600
    ):
        raise ValueError("display_cooldown_seconds must be between 0 and 3600")

    inserted = connection.execute(
        """
        INSERT INTO hiperwall_display_states (
            rule_id, source_name, state,
            active_rule_event_id, open_action_id, instance_id,
            cooldown_seconds, displaying_since, created_at, updated_at
        ) VALUES (?, ?, 'DISPLAYING', ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (rule_id, source_name) DO NOTHING
        """,
        (
            event.rule_id,
            source_name,
            event.id,
            action.id,
            instance_id.strip(),
            float(cooldown_seconds),
            now_text,
            now_text,
            now_text,
        ),
    )
    if inserted.rowcount == 1:
        return True
    updated = connection.execute(
        """
        UPDATE hiperwall_display_states
        SET state = 'DISPLAYING',
            active_rule_event_id = ?,
            open_action_id = ?,
            close_action_id = NULL,
            instance_id = ?,
            cooldown_seconds = ?,
            displaying_since = ?,
            opened_at = NULL,
            close_succeeded_at = NULL,
            cooldown_until = NULL,
            updated_at = ?
        WHERE rule_id = ? AND source_name = ? AND state = 'IDLE'
        """,
        (
            event.id,
            action.id,
            instance_id.strip(),
            float(cooldown_seconds),
            now_text,
            now_text,
            event.rule_id,
            source_name,
        ),
    )
    return updated.rowcount == 1


def _record_from_row(row: sqlite3.Row) -> DisplayStateRecord:
    return DisplayStateRecord(
        rule_id=str(row["rule_id"]),
        source_name=str(row["source_name"]),
        state=str(row["state"]),
        active_rule_event_id=_optional_text(row["active_rule_event_id"]),
        open_action_id=_optional_text(row["open_action_id"]),
        close_action_id=_optional_text(row["close_action_id"]),
        instance_id=_optional_text(row["instance_id"]),
        cooldown_seconds=float(row["cooldown_seconds"]),
        displaying_since=_optional_datetime(row["displaying_since"]),
        opened_at=_optional_datetime(row["opened_at"]),
        close_succeeded_at=_optional_datetime(row["close_succeeded_at"]),
        cooldown_until=_optional_datetime(row["cooldown_until"]),
        last_rule_event_id=_optional_text(row["last_rule_event_id"]),
        last_open_action_id=_optional_text(row["last_open_action_id"]),
        last_close_action_id=_optional_text(row["last_close_action_id"]),
        last_closed_at=_optional_datetime(row["last_closed_at"]),
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_datetime(value: object) -> datetime | None:
    return _datetime_from_text(str(value)) if value is not None else None


def _datetime_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _utc_datetime(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


def _utc_text(value: datetime | None = None) -> str:
    return _utc_datetime(value).isoformat()


__all__ = [
    "DisplayGateResult",
    "DisplayStateRecord",
    "DisplayStateRepository",
    "gate_live_display_events",
    "record_display_action_success",
    "release_expired_display_states",
    "release_failed_open_state",
]
