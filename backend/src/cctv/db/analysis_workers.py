"""Durable analysis-worker recovery and append-only failure history."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cctv.core.logging import redact_text
from cctv.db.database import connect_database


@dataclass(frozen=True, slots=True)
class AnalysisRunRecoveryResult:
    recovered_run_ids: tuple[str, ...]
    recovered_run_count: int
    closed_active_track_count: int
    preserved_person_instance_count: int
    recovered_at: datetime
    reason: str


@dataclass(frozen=True, slots=True)
class AnalysisWorkerFailureRecord:
    id: int
    camera_id: int
    analysis_run_id: str | None
    owner_id: str
    process_instance_id: str
    failure_type: str
    failure_message: str
    failure_at: datetime
    restart_count: int
    created_at: datetime


class AnalysisRunRecoveryRepository:
    """Finish only running analyses that have no valid run-bound lease."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def recover_orphaned_runs(
        self,
        *,
        now: datetime | None = None,
        reason: str = "recovered_on_startup",
        excluded_run_ids: tuple[str, ...] = (),
    ) -> AnalysisRunRecoveryResult:
        recovered_at = _utc(now)
        recovered_text = _utc_text(recovered_at)
        normalized_reason = _bounded(reason, 128, "reason")
        exclusions = tuple(dict.fromkeys(item.strip() for item in excluded_run_ids if item.strip()))
        exclusion_sql = ""
        parameters: list[object] = [recovered_text]
        if exclusions:
            placeholders = ", ".join("?" for _ in exclusions)
            exclusion_sql = f"AND run.id NOT IN ({placeholders})"
            parameters.extend(exclusions)

        with closing(connect_database(self.database_path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    f"""
                    SELECT run.id
                    FROM analysis_runs AS run
                    LEFT JOIN analysis_worker_leases AS lease
                        ON lease.analysis_run_id = run.id
                        AND lease.expires_at > ?
                    WHERE run.status = 'running'
                        AND lease.camera_id IS NULL
                        {exclusion_sql}
                    ORDER BY run.started_at, run.id
                    """,
                    parameters,
                ).fetchall()
                run_ids = tuple(str(row["id"]) for row in rows)
                closed_tracks, person_instances, _ = _interrupt_runs(
                    connection,
                    run_ids=run_ids,
                    completed_at=recovered_text,
                    reason=normalized_reason,
                    include_stopped=False,
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return AnalysisRunRecoveryResult(
            recovered_run_ids=run_ids,
            recovered_run_count=len(run_ids),
            closed_active_track_count=closed_tracks,
            preserved_person_instance_count=person_instances,
            recovered_at=recovered_at,
            reason=normalized_reason,
        )

    def interrupt_run(
        self,
        analysis_run_id: str,
        *,
        reason: str,
        now: datetime | None = None,
        include_stopped: bool = False,
    ) -> bool:
        run_id = _bounded(analysis_run_id, 128, "analysis_run_id")
        interrupted_at = _utc(now)
        with closing(connect_database(self.database_path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = _interrupt_runs(
                    connection,
                    run_ids=(run_id,),
                    completed_at=_utc_text(interrupted_at),
                    reason=_bounded(reason, 128, "reason"),
                    include_stopped=include_stopped,
                )[2]
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return cursor > 0


class AnalysisWorkerFailureRepository:
    """Append and query sanitized worker failures across Backend restarts."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def record(
        self,
        *,
        camera_id: int,
        analysis_run_id: str | None,
        owner_id: str,
        process_instance_id: str,
        failure_type: str,
        failure_message: str,
        failure_at: datetime,
        restart_count: int,
    ) -> AnalysisWorkerFailureRecord:
        if camera_id < 1:
            raise ValueError("camera_id must be at least 1")
        if restart_count < 1:
            raise ValueError("restart_count must be at least 1")
        failure_time = _utc(failure_at)
        with closing(connect_database(self.database_path)) as connection, connection:
            stored_run_id = None
            if analysis_run_id is not None:
                candidate = _bounded(analysis_run_id, 128, "analysis_run_id")
                exists = connection.execute(
                    "SELECT 1 FROM analysis_runs WHERE id = ?",
                    (candidate,),
                ).fetchone()
                stored_run_id = candidate if exists is not None else None
            connection.execute(
                """
                INSERT INTO analysis_worker_failures (
                    camera_id, analysis_run_id, owner_id, process_instance_id,
                    failure_type, failure_message, failure_at, restart_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(process_instance_id) DO NOTHING
                """,
                (
                    camera_id,
                    stored_run_id,
                    _bounded(owner_id, 128, "owner_id"),
                    _bounded(process_instance_id, 128, "process_instance_id"),
                    _bounded(failure_type, 128, "failure_type"),
                    _failure_message(failure_message),
                    _utc_text(failure_time),
                    restart_count,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM analysis_worker_failures
                WHERE process_instance_id = ?
                """,
                (process_instance_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("analysis worker failure could not be read")
        return _failure_from_row(row)

    def latest(self, camera_id: int) -> AnalysisWorkerFailureRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT * FROM analysis_worker_failures
                WHERE camera_id = ?
                ORDER BY failure_at DESC, id DESC
                LIMIT 1
                """,
                (camera_id,),
            ).fetchone()
        return _failure_from_row(row) if row is not None else None

    def list_since(
        self,
        camera_id: int,
        *,
        since: datetime,
    ) -> tuple[AnalysisWorkerFailureRecord, ...]:
        with closing(connect_database(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT * FROM analysis_worker_failures
                WHERE camera_id = ? AND failure_at >= ?
                ORDER BY failure_at, id
                """,
                (camera_id, _utc_text(_utc(since))),
            ).fetchall()
        return tuple(_failure_from_row(row) for row in rows)


def _interrupt_runs(
    connection: sqlite3.Connection,
    *,
    run_ids: tuple[str, ...],
    completed_at: str,
    reason: str,
    include_stopped: bool,
) -> tuple[int, int, int]:
    if not run_ids:
        return 0, 0, 0
    placeholders = ", ".join("?" for _ in run_ids)
    statuses = "('running', 'stopped')" if include_stopped else "('running')"
    active_tracks = int(
        connection.execute(
            f"""
            SELECT COUNT(*) FROM tracks
            WHERE analysis_run_id IN ({placeholders}) AND is_active = 1
            """,
            run_ids,
        ).fetchone()[0]
    )
    person_instances = int(
        connection.execute(
            f"""
            SELECT COUNT(*) FROM person_instances
            WHERE analysis_run_id IN ({placeholders})
            """,
            run_ids,
        ).fetchone()[0]
    )
    cursor = connection.execute(
        f"""
        UPDATE analysis_runs
        SET status = 'interrupted', completed_at = ?,
            processed_frames = (
                SELECT COUNT(*) FROM analyzed_frames AS frame
                WHERE frame.analysis_run_id = analysis_runs.id
            ),
            total_detections = (
                SELECT COUNT(*)
                FROM detections AS detection
                JOIN analyzed_frames AS frame
                    ON frame.id = detection.analyzed_frame_id
                WHERE frame.analysis_run_id = analysis_runs.id
            ),
            error_type = NULL,
            completion_reason = ?
        WHERE id IN ({placeholders}) AND status IN {statuses}
        """,
        (completed_at, reason, *run_ids),
    )
    if cursor.rowcount:
        connection.execute(
            f"""
            UPDATE tracks
            SET is_active = 0, updated_at = ?
            WHERE analysis_run_id IN ({placeholders}) AND is_active = 1
            """,
            (completed_at, *run_ids),
        )
    return active_tracks, person_instances, cursor.rowcount


def _failure_message(value: str) -> str:
    sanitized = redact_text(str(value)).strip() or "unknown worker failure"
    return sanitized[:1024]


def _bounded(value: str, maximum: int, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized[:maximum]


def _failure_from_row(row: sqlite3.Row) -> AnalysisWorkerFailureRecord:
    return AnalysisWorkerFailureRecord(
        id=int(row["id"]),
        camera_id=int(row["camera_id"]),
        analysis_run_id=(
            str(row["analysis_run_id"]) if row["analysis_run_id"] is not None else None
        ),
        owner_id=str(row["owner_id"]),
        process_instance_id=str(row["process_instance_id"]),
        failure_type=str(row["failure_type"]),
        failure_message=str(row["failure_message"]),
        failure_at=_datetime(str(row["failure_at"])),
        restart_count=int(row["restart_count"]),
        created_at=_datetime(str(row["created_at"])),
    )


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("timestamps must include timezone information")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


__all__ = [
    "AnalysisRunRecoveryRepository",
    "AnalysisRunRecoveryResult",
    "AnalysisWorkerFailureRecord",
    "AnalysisWorkerFailureRepository",
]
