"""SQLite repository for simulated Hiperwall display actions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cctv.db.database import connect_database
from cctv.db.detections import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

if TYPE_CHECKING:
    from cctv.hiperwall import DisplayAction


@dataclass(frozen=True, slots=True)
class DisplayActionRecord:
    """One simulated action joined with its rule-event and analysis context."""

    id: str
    rule_event_id: str
    rule_id: str
    analysis_run_id: str
    source_name: str
    track_id: int
    event_type: str
    event_state: str
    action_type: str
    mode: str
    status: str
    request: dict[str, Any]
    result: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DisplayActionPage:
    items: tuple[DisplayActionRecord, ...]
    total: int


class DisplayActionRepository:
    """Query Hiperwall operations without exposing credentials or filesystem paths."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def get(self, action_id: str) -> DisplayActionRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                f"{_SELECT_ACTIONS} WHERE action.id = ?",
                (action_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list(
        self,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        rule_event_id: str | None = None,
        analysis_run_id: str | None = None,
        source_name: str | None = None,
        action_type: str | None = None,
        status: str | None = None,
    ) -> DisplayActionPage:
        _validate_pagination(page, limit)
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("action.rule_event_id", rule_event_id),
            ("event.analysis_run_id", analysis_run_id),
            ("run.source_name", source_name),
            ("action.action_type", action_type),
            ("action.status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(_required_text(value, column))
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) {_JOIN_ACTIONS}{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                {_SELECT_ACTIONS}{where_sql}
                ORDER BY action.created_at DESC, action.id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return DisplayActionPage(
            items=tuple(_record_from_row(row) for row in rows),
            total=total,
        )


_JOIN_ACTIONS = """
    FROM display_actions AS action
    JOIN rule_events AS event ON event.id = action.rule_event_id
    JOIN analysis_runs AS run ON run.id = event.analysis_run_id
"""

_SELECT_ACTIONS = f"""
    SELECT
        action.*,
        event.rule_id,
        event.analysis_run_id,
        event.track_id,
        event.event_type,
        event.event_state,
        run.source_name
    {_JOIN_ACTIONS}
"""


def insert_display_actions(
    connection: sqlite3.Connection,
    *,
    actions: Collection[DisplayAction],
) -> None:
    """Insert action plans inside the same transaction as their rule events."""
    connection.executemany(
        """
        INSERT INTO display_actions (
            id, rule_event_id, action_type, mode, status, request_json, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                action.id,
                action.rule_event_id,
                action.action_type,
                action.mode,
                action.status,
                _json_object(action.request, "display action request"),
                _json_object(action.result, "display action result"),
            )
            for action in actions
        ),
    )


def _required_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _json_object(value: dict[str, Any], name: str) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JSON-compatible values") from error


def _json_from_text(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("stored display action JSON must be an object")
    return parsed


def _datetime_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _record_from_row(row: sqlite3.Row) -> DisplayActionRecord:
    return DisplayActionRecord(
        id=str(row["id"]),
        rule_event_id=str(row["rule_event_id"]),
        rule_id=str(row["rule_id"]),
        analysis_run_id=str(row["analysis_run_id"]),
        source_name=str(row["source_name"]),
        track_id=int(row["track_id"]),
        event_type=str(row["event_type"]),
        event_state=str(row["event_state"]),
        action_type=str(row["action_type"]),
        mode=str(row["mode"]),
        status=str(row["status"]),
        request=_json_from_text(str(row["request_json"])),
        result=_json_from_text(str(row["result_json"])),
        created_at=_datetime_from_text(str(row["created_at"])),
    )


def _validate_pagination(page: int, limit: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


__all__ = [
    "DisplayActionPage",
    "DisplayActionRecord",
    "DisplayActionRepository",
    "insert_display_actions",
]
