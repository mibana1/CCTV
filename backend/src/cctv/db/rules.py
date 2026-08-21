"""SQLite repository for generic rule configuration and emitted events."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cctv.db.database import connect_database
from cctv.rules import RuleDefinition

if TYPE_CHECKING:
    from cctv.rules import RuleEvent


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """One stored evaluator-neutral rule configuration."""

    id: str
    name: str
    source_name: str
    rule_type: str
    class_name: str | None
    geometry: dict[str, Any]
    parameters: dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def to_definition(self) -> RuleDefinition:
        return RuleDefinition(
            id=self.id,
            name=self.name,
            source_name=self.source_name,
            rule_type=self.rule_type,
            class_name=self.class_name,
            geometry=self.geometry,
            parameters=self.parameters,
            enabled=self.enabled,
        )


@dataclass(frozen=True, slots=True)
class RuleEventRecord:
    """One persisted rule event joined to its rule name and type."""

    id: str
    rule_id: str
    rule_name: str
    rule_type: str
    analysis_run_id: str
    track_id: int
    analyzed_frame_id: int
    event_type: str
    event_state: str
    occurred_at_seconds: float
    class_name: str
    confidence: float
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RulePage:
    items: tuple[RuleRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class RuleEventPage:
    items: tuple[RuleEventRecord, ...]
    total: int


class RuleRepository:
    """Create, enable, list, and query generic rule configurations and events."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def create_rule(
        self,
        *,
        name: str,
        source_name: str,
        rule_type: str,
        geometry: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        class_name: str | None = None,
        enabled: bool = True,
        rule_id: str | None = None,
    ) -> RuleRecord:
        """Persist a JSON-configured rule without coupling SQL to evaluator types."""
        normalized_name = _required_text(name, "name")
        normalized_source = _required_text(source_name, "source_name")
        normalized_type = _required_text(rule_type, "rule_type")
        normalized_class = class_name.strip() if class_name is not None else None
        if normalized_class == "":
            raise ValueError("class_name must not be empty")
        geometry_json = _json_object(geometry, "geometry")
        parameters_json = _json_object(parameters or {}, "parameters")
        identifier = rule_id or str(uuid4())

        with closing(connect_database(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO rules (
                    id, name, source_name, rule_type, class_name,
                    geometry_json, parameters_json, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    normalized_name,
                    normalized_source,
                    normalized_type,
                    normalized_class,
                    geometry_json,
                    parameters_json,
                    int(enabled),
                ),
            )
            row = connection.execute("SELECT * FROM rules WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise RuntimeError("rule was not created")
        return _rule_from_row(row)

    def get_rule(self, rule_id: str) -> RuleRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM rules WHERE id = ? AND deleted_at IS NULL",
                (rule_id,),
            ).fetchone()
        return _rule_from_row(row) if row is not None else None

    def update_rule(
        self,
        rule_id: str,
        *,
        name: str,
        source_name: str,
        rule_type: str,
        geometry: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        class_name: str | None = None,
        enabled: bool = True,
    ) -> RuleRecord:
        """Replace one visible rule configuration while retaining its identity."""
        normalized_name = _required_text(name, "name")
        normalized_source = _required_text(source_name, "source_name")
        normalized_type = _required_text(rule_type, "rule_type")
        normalized_class = class_name.strip() if class_name is not None else None
        if normalized_class == "":
            raise ValueError("class_name must not be empty")
        geometry_json = _json_object(geometry, "geometry")
        parameters_json = _json_object(parameters or {}, "parameters")
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE rules
                SET name = ?, source_name = ?, rule_type = ?, class_name = ?,
                    geometry_json = ?, parameters_json = ?, enabled = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted_at IS NULL
                """,
                (
                    normalized_name,
                    normalized_source,
                    normalized_type,
                    normalized_class,
                    geometry_json,
                    parameters_json,
                    int(enabled),
                    rule_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"rule does not exist: {rule_id}")
            row = connection.execute(
                "SELECT * FROM rules WHERE id = ? AND deleted_at IS NULL",
                (rule_id,),
            ).fetchone()
        if row is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("updated rule could not be read")
        return _rule_from_row(row)

    def delete_rule(self, rule_id: str) -> None:
        """Hide a rule while retaining its events and durable display actions."""
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE rules
                SET enabled = 0,
                    source_name = '__deleted__/' || id,
                    deleted_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted_at IS NULL
                """,
                (rule_id,),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"rule does not exist: {rule_id}")

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> RuleRecord:
        """Enable or disable one rule without changing its geometry contract."""
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE rules
                SET enabled = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted_at IS NULL
                """,
                (int(enabled), rule_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"rule does not exist: {rule_id}")
            row = connection.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
        if row is None:
            raise RuntimeError("updated rule could not be read")
        return _rule_from_row(row)

    def set_rule_hiperwall_mapping(
        self,
        rule_id: str,
        mapping: dict[str, Any] | None,
    ) -> RuleRecord:
        """Add, replace, or remove the Hiperwall mapping inside rule parameters."""
        with closing(connect_database(self.database_path)) as connection, connection:
            row = connection.execute(
                "SELECT parameters_json FROM rules WHERE id = ? AND deleted_at IS NULL",
                (rule_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"rule does not exist: {rule_id}")
            parameters = _json_from_text(str(row["parameters_json"]))
            if mapping is None:
                parameters.pop("hiperwall", None)
            else:
                parameters["hiperwall"] = mapping
            connection.execute(
                """
                UPDATE rules
                SET parameters_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted_at IS NULL
                """,
                (_json_object(parameters, "parameters"), rule_id),
            )
            updated = connection.execute(
                "SELECT * FROM rules WHERE id = ?",
                (rule_id,),
            ).fetchone()
        if updated is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("updated rule could not be read")
        return _rule_from_row(updated)

    def list_rules(
        self,
        *,
        page: int = 1,
        limit: int = 50,
        source_name: str | None = None,
        rule_type: str | None = None,
        enabled: bool | None = None,
    ) -> RulePage:
        _validate_pagination(page, limit)
        clauses: list[str] = ["deleted_at IS NULL"]
        parameters: list[object] = []
        for column, value in (("source_name", source_name), ("rule_type", rule_type)):
            if value is not None:
                normalized = _required_text(value, column)
                clauses.append(f"{column} = ? COLLATE NOCASE")
                parameters.append(normalized)
        if enabled is not None:
            clauses.append("enabled = ?")
            parameters.append(int(enabled))
        where_sql = f" WHERE {' AND '.join(clauses)}"
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(f"SELECT COUNT(*) FROM rules{where_sql}", parameters).fetchone()[
                    0
                ]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM rules{where_sql}
                ORDER BY source_name, name, id
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return RulePage(items=tuple(_rule_from_row(row) for row in rows), total=total)

    def list_rule_definitions(
        self,
        *,
        source_name: str,
        enabled: bool = True,
    ) -> tuple[RuleDefinition, ...]:
        """Load all matching definitions once when an analysis worker starts."""
        page = self.list_rules(
            page=1,
            limit=100,
            source_name=source_name,
            enabled=enabled,
        )
        if page.total > 100:
            raise RuntimeError("a source cannot load more than 100 rules")
        return tuple(record.to_definition() for record in page.items)

    def list_events(
        self,
        *,
        page: int = 1,
        limit: int = 50,
        rule_id: str | None = None,
        analysis_run_id: str | None = None,
        track_id: int | None = None,
        event_type: str | None = None,
        event_state: str | None = None,
        occurred_from_seconds: float | None = None,
        occurred_to_seconds: float | None = None,
    ) -> RuleEventPage:
        _validate_pagination(page, limit)
        if track_id is not None and track_id < 1:
            raise ValueError("track_id must be at least 1")
        for name, value in (
            ("occurred_from_seconds", occurred_from_seconds),
            ("occurred_to_seconds", occurred_to_seconds),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite number that is zero or greater")
        if (
            occurred_from_seconds is not None
            and occurred_to_seconds is not None
            and occurred_from_seconds > occurred_to_seconds
        ):
            raise ValueError(
                "occurred_from_seconds must be less than or equal to occurred_to_seconds"
            )
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("event.rule_id", rule_id),
            ("event.analysis_run_id", analysis_run_id),
            ("event.event_type", event_type),
            ("event.event_state", event_state),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(_required_text(value, column))
        if track_id is not None:
            clauses.append("event.track_id = ?")
            parameters.append(track_id)
        if occurred_from_seconds is not None:
            clauses.append("event.occurred_at_seconds >= ?")
            parameters.append(occurred_from_seconds)
        if occurred_to_seconds is not None:
            clauses.append("event.occurred_at_seconds <= ?")
            parameters.append(occurred_to_seconds)
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        join_sql = """
            FROM rule_events AS event
            JOIN rules AS rule ON rule.id = event.rule_id
        """
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(f"SELECT COUNT(*) {join_sql}{where_sql}", parameters).fetchone()[
                    0
                ]
            )
            rows = connection.execute(
                f"""
                SELECT event.*, rule.name AS rule_name, rule.rule_type
                {join_sql}{where_sql}
                ORDER BY event.created_at DESC, event.id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return RuleEventPage(
            items=tuple(_event_from_row(row) for row in rows),
            total=total,
        )


def insert_rule_events(
    connection: sqlite3.Connection,
    *,
    analysis_run_id: str,
    analyzed_frame_id: int,
    events: Collection[RuleEvent],
) -> None:
    """Insert emitted events inside the analyzed-frame transaction."""
    connection.executemany(
        """
        INSERT INTO rule_events (
            id, rule_id, analysis_run_id, track_id, analyzed_frame_id,
            event_type, event_state, occurred_at_seconds,
            class_name, confidence, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                event.id,
                event.rule_id,
                analysis_run_id,
                event.track_id,
                analyzed_frame_id,
                event.event_type,
                event.event_state,
                event.occurred_at_seconds,
                event.class_name,
                event.confidence,
                _json_object(event.payload, "event payload"),
            )
            for event in events
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
        raise TypeError("stored rule JSON must be an object")
    return parsed


def _datetime_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _rule_from_row(row: sqlite3.Row) -> RuleRecord:
    return RuleRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        source_name=str(row["source_name"]),
        rule_type=str(row["rule_type"]),
        class_name=str(row["class_name"]) if row["class_name"] is not None else None,
        geometry=_json_from_text(str(row["geometry_json"])),
        parameters=_json_from_text(str(row["parameters_json"])),
        enabled=bool(row["enabled"]),
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _event_from_row(row: sqlite3.Row) -> RuleEventRecord:
    return RuleEventRecord(
        id=str(row["id"]),
        rule_id=str(row["rule_id"]),
        rule_name=str(row["rule_name"]),
        rule_type=str(row["rule_type"]),
        analysis_run_id=str(row["analysis_run_id"]),
        track_id=int(row["track_id"]),
        analyzed_frame_id=int(row["analyzed_frame_id"]),
        event_type=str(row["event_type"]),
        event_state=str(row["event_state"]),
        occurred_at_seconds=float(row["occurred_at_seconds"]),
        class_name=str(row["class_name"]),
        confidence=float(row["confidence"]),
        payload=_json_from_text(str(row["payload_json"])),
        created_at=_datetime_from_text(str(row["created_at"])),
    )


def _validate_pagination(page: int, limit: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")


__all__ = [
    "RuleEventPage",
    "RuleEventRecord",
    "RulePage",
    "RuleRecord",
    "RuleRepository",
    "insert_rule_events",
]
