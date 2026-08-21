"""Add durable rule/source Hiperwall display lifecycle state."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from sqlite3 import Connection
from typing import Any


def apply(connection: Connection) -> None:
    connection.execute(
        """
        CREATE TABLE hiperwall_display_states (
            rule_id TEXT NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
            source_name TEXT NOT NULL CHECK (length(source_name) > 0),
            state TEXT NOT NULL
                CHECK (state IN ('IDLE', 'DISPLAYING', 'COOLDOWN')),
            active_rule_event_id TEXT,
            open_action_id TEXT,
            close_action_id TEXT,
            instance_id TEXT,
            cooldown_seconds REAL NOT NULL DEFAULT 0
                CHECK (cooldown_seconds >= 0 AND cooldown_seconds <= 3600),
            displaying_since TEXT,
            opened_at TEXT,
            close_succeeded_at TEXT,
            cooldown_until TEXT,
            last_rule_event_id TEXT,
            last_open_action_id TEXT,
            last_close_action_id TEXT,
            last_closed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (rule_id, source_name)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_hiperwall_display_states_state_cooldown
        ON hiperwall_display_states (state, cooldown_until, rule_id, source_name)
        """
    )
    _adopt_outstanding_live_actions(connection)


def _adopt_outstanding_live_actions(connection: Connection) -> None:
    """Collapse pre-gate open retries and adopt one safe lifecycle per rule/source."""
    now = datetime.now(UTC).isoformat()
    outstanding_closes = connection.execute(
        """
        SELECT
            action.id, action.rule_event_id, action.request_json,
            action.created_at, action.available_at,
            event.rule_id, run.source_name, rule.parameters_json,
            opened.id AS open_action_id, opened.completed_at AS opened_at
        FROM display_actions AS action
        JOIN rule_events AS event ON event.id = action.rule_event_id
        JOIN analysis_runs AS run ON run.id = event.analysis_run_id
        JOIN rules AS rule ON rule.id = event.rule_id
        LEFT JOIN display_actions AS opened
            ON opened.rule_event_id = action.rule_event_id
            AND opened.action_type = 'open_source'
            AND opened.status = 'succeeded'
        WHERE action.mode = 'live'
            AND action.action_type = 'restore_layout'
            AND action.status IN ('pending', 'processing', 'retry')
        ORDER BY action.available_at DESC, action.created_at DESC, action.id DESC
        """
    ).fetchall()
    close_by_key: dict[tuple[str, str], Any] = {}
    for row in outstanding_closes:
        key = (str(row["rule_id"]), str(row["source_name"]))
        close_by_key.setdefault(key, row)

    outstanding_opens = connection.execute(
        """
        SELECT
            action.id, action.rule_event_id, action.request_json,
            action.created_at, action.available_at,
            event.rule_id, run.source_name, rule.parameters_json
        FROM display_actions AS action
        JOIN rule_events AS event ON event.id = action.rule_event_id
        JOIN analysis_runs AS run ON run.id = event.analysis_run_id
        JOIN rules AS rule ON rule.id = event.rule_id
        WHERE action.mode = 'live'
            AND action.action_type = 'open_source'
            AND action.status IN ('pending', 'processing', 'retry')
        ORDER BY action.created_at DESC, action.id DESC
        """
    ).fetchall()
    opens_by_key: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in outstanding_opens:
        opens_by_key[(str(row["rule_id"]), str(row["source_name"]))].append(row)

    for key, close in close_by_key.items():
        _seed_state_from_close(connection, close, now=now)
        _fail_superseded_opens(connection, opens_by_key.pop(key, ()), keep_id=None, now=now)

    for rows in opens_by_key.values():
        keep = rows[0]
        _seed_state_from_open(connection, keep, now=now)
        _fail_superseded_opens(connection, rows, keep_id=str(keep["id"]), now=now)


def _seed_state_from_open(connection: Connection, row: Any, *, now: str) -> None:
    request = _json_object(row["request_json"])
    connection.execute(
        """
        INSERT INTO hiperwall_display_states (
            rule_id, source_name, state,
            active_rule_event_id, open_action_id, instance_id,
            cooldown_seconds, displaying_since, created_at, updated_at
        ) VALUES (?, ?, 'DISPLAYING', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["rule_id"]),
            str(row["source_name"]),
            str(row["rule_event_id"]),
            str(row["id"]),
            _instance_id(request, str(row["id"])),
            _cooldown_seconds(row["parameters_json"]),
            str(row["created_at"]),
            now,
            now,
        ),
    )


def _seed_state_from_close(connection: Connection, row: Any, *, now: str) -> None:
    request = _json_object(row["request_json"])
    opened_at = str(row["opened_at"]) if row["opened_at"] is not None else None
    connection.execute(
        """
        INSERT INTO hiperwall_display_states (
            rule_id, source_name, state,
            active_rule_event_id, open_action_id, close_action_id, instance_id,
            cooldown_seconds, displaying_since, opened_at, created_at, updated_at
        ) VALUES (?, ?, 'DISPLAYING', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["rule_id"]),
            str(row["source_name"]),
            str(row["rule_event_id"]),
            str(row["open_action_id"] or f"legacy-open-{row['rule_event_id']}"),
            str(row["id"]),
            _instance_id(request, str(row["id"])),
            _cooldown_seconds(row["parameters_json"]),
            opened_at or str(row["created_at"]),
            opened_at,
            now,
            now,
        ),
    )


def _fail_superseded_opens(
    connection: Connection,
    rows: Any,
    *,
    keep_id: str | None,
    now: str,
) -> None:
    for row in rows:
        action_id = str(row["id"])
        if action_id == keep_id:
            continue
        connection.execute(
            """
            UPDATE display_actions
            SET status = 'failed', completed_at = ?,
                last_error_code = 'superseded_by_display_gate',
                last_error_message =
                    'Superseded while enabling the rule/camera display lifecycle gate',
                updated_at = ?
            WHERE id = ? AND status IN ('pending', 'processing', 'retry')
            """,
            (now, now, action_id),
        )


def _cooldown_seconds(parameters_json: object) -> float:
    parameters = _json_object(parameters_json)
    value = parameters.get("cooldown_seconds", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return min(max(float(value), 0), 3_600)


def _instance_id(request: dict[str, Any], action_id: str) -> str:
    value = request.get("instance_id")
    return value.strip() if isinstance(value, str) and value.strip() else f"legacy-{action_id}"


def _json_object(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
