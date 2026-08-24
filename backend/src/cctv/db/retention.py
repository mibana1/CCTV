"""Batched retention queries that preserve event and Hiperwall audit history."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection

from cctv.db.database import connect_database


@dataclass(frozen=True, slots=True)
class RetentionCounts:
    analyzed_frames: int = 0
    detections: int = 0
    track_observations: int = 0
    tracks: int = 0
    rule_events: int = 0
    display_actions: int = 0
    snapshots: int = 0
    snapshot_bytes: int = 0

    def __add__(self, other: RetentionCounts) -> RetentionCounts:
        return RetentionCounts(
            analyzed_frames=self.analyzed_frames + other.analyzed_frames,
            detections=self.detections + other.detections,
            track_observations=self.track_observations + other.track_observations,
            tracks=self.tracks + other.tracks,
            rule_events=self.rule_events + other.rule_events,
            display_actions=self.display_actions + other.display_actions,
            snapshots=self.snapshots + other.snapshots,
            snapshot_bytes=self.snapshot_bytes + other.snapshot_bytes,
        )


@dataclass(frozen=True, slots=True)
class RetentionDatabaseMetrics:
    database_bytes: int
    wal_bytes: int
    shm_bytes: int
    page_count: int
    page_size: int
    freelist_count: int

    @property
    def total_bytes(self) -> int:
        return self.database_bytes + self.wal_bytes + self.shm_bytes


@dataclass(frozen=True, slots=True)
class DatabaseActivity:
    running_analysis_runs: int
    active_analysis_leases: int
    processing_display_actions: int

    @property
    def active(self) -> bool:
        return any(
            (
                self.running_analysis_runs,
                self.active_analysis_leases,
                self.processing_display_actions,
            )
        )


@dataclass(frozen=True, slots=True)
class WalCheckpointResult:
    mode: str
    busy: int
    log_frames: int
    checkpointed_frames: int


class RetentionRepository:
    """Preview and delete bounded sets of expired analysis records."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def metrics(self) -> RetentionDatabaseMetrics:
        with closing(connect_database(self.database_path)) as connection:
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            freelist_count = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        return RetentionDatabaseMetrics(
            database_bytes=_file_size(self.database_path),
            wal_bytes=_file_size(Path(f"{self.database_path}-wal")),
            shm_bytes=_file_size(Path(f"{self.database_path}-shm")),
            page_count=page_count,
            page_size=page_size,
            freelist_count=freelist_count,
        )

    def activity(self, *, now: datetime) -> DatabaseActivity:
        now_text = _sqlite_utc_text(now)
        with closing(connect_database(self.database_path)) as connection:
            running_analysis_runs = _count(
                connection,
                "SELECT COUNT(*) FROM analysis_runs WHERE status = 'running'",
                (),
            )
            active_analysis_leases = _count(
                connection,
                "SELECT COUNT(*) FROM analysis_worker_leases WHERE expires_at > ?",
                (now_text,),
            )
            processing_display_actions = _count(
                connection,
                "SELECT COUNT(*) FROM display_actions WHERE status = 'processing'",
                (),
            )
        return DatabaseActivity(
            running_analysis_runs=running_analysis_runs,
            active_analysis_leases=active_analysis_leases,
            processing_display_actions=processing_display_actions,
        )

    def checkpoint(self, *, mode: str = "passive") -> WalCheckpointResult:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"passive", "truncate"}:
            raise ValueError("checkpoint mode must be passive or truncate")
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                f"PRAGMA wal_checkpoint({normalized_mode.upper()})"
            ).fetchone()
        return WalCheckpointResult(
            mode=normalized_mode,
            busy=int(row[0]),
            log_frames=int(row[1]),
            checkpointed_frames=int(row[2]),
        )

    def online_backup(self, destination: str | Path) -> Path:
        final_path = Path(destination).expanduser().resolve()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise FileExistsError(f"retention backup already exists: {final_path}")
        partial_path = final_path.with_suffix(f"{final_path.suffix}.partial")
        if partial_path.exists():
            raise FileExistsError(f"partial retention backup already exists: {partial_path}")

        completed = False
        try:
            with (
                closing(connect_database(self.database_path)) as source,
                closing(sqlite3.connect(partial_path)) as target,
            ):
                source.backup(target)
                target.commit()
                integrity_check = target.execute("PRAGMA integrity_check").fetchall()
                if len(integrity_check) != 1 or str(integrity_check[0][0]).lower() != "ok":
                    raise RuntimeError("online retention backup failed SQLite integrity_check")
            partial_path.replace(final_path)
            completed = True
            return final_path
        finally:
            if not completed:
                partial_path.unlink(missing_ok=True)

    def vacuum(self) -> None:
        with closing(connect_database(self.database_path)) as connection:
            connection.execute("VACUUM")

    def preview(
        self,
        *,
        frame_cutoff: datetime,
        audit_cutoff: datetime,
        limit: int,
    ) -> RetentionCounts:
        _validate_limit(limit)
        with closing(connect_database(self.database_path)) as connection:
            frame_ids = _eligible_frame_ids(
                connection,
                frame_cutoff=frame_cutoff,
                audit_cutoff=audit_cutoff,
                limit=limit,
            )
            counts = _frame_counts(connection, frame_ids)
            orphan_tracks = _preview_orphan_track_keys(
                connection,
                frame_ids=frame_ids,
                frame_cutoff=frame_cutoff,
                limit=limit,
            )
        return counts + RetentionCounts(tracks=len(orphan_tracks))

    def delete_batch(
        self,
        *,
        frame_cutoff: datetime,
        audit_cutoff: datetime,
        batch_size: int,
    ) -> RetentionCounts:
        _validate_limit(batch_size)
        with closing(connect_database(self.database_path)) as connection, connection:
            frame_ids = _eligible_frame_ids(
                connection,
                frame_cutoff=frame_cutoff,
                audit_cutoff=audit_cutoff,
                limit=batch_size,
            )
            counts = _frame_counts(connection, frame_ids)
            if frame_ids:
                placeholders = ",".join("?" for _ in frame_ids)
                connection.execute(
                    f"DELETE FROM analyzed_frames WHERE id IN ({placeholders})",
                    frame_ids,
                )

            orphan_tracks = _orphan_track_keys(
                connection,
                frame_cutoff=frame_cutoff,
                limit=batch_size,
            )
            if orphan_tracks:
                connection.executemany(
                    "DELETE FROM tracks WHERE analysis_run_id = ? AND track_id = ?",
                    orphan_tracks,
                )
        return counts + RetentionCounts(tracks=len(orphan_tracks))


def _eligible_frame_ids(
    connection: Connection,
    *,
    frame_cutoff: datetime,
    audit_cutoff: datetime,
    limit: int,
) -> tuple[int, ...]:
    rows = connection.execute(
        """
        SELECT frame.id
        FROM analyzed_frames AS frame
        WHERE (
            frame.created_at < ?
            AND NOT EXISTS (
                SELECT 1 FROM rule_events AS event
                WHERE event.analyzed_frame_id = frame.id
            )
        ) OR (
            EXISTS (
                SELECT 1 FROM rule_events AS event
                WHERE event.analyzed_frame_id = frame.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM rule_events AS event
                WHERE event.analyzed_frame_id = frame.id
                    AND event.created_at >= ?
            )
            AND NOT EXISTS (
                SELECT 1
                FROM rule_events AS event
                JOIN display_actions AS action ON action.rule_event_id = event.id
                WHERE event.analyzed_frame_id = frame.id
                    AND action.created_at >= ?
            )
            AND NOT EXISTS (
                SELECT 1
                FROM rule_events AS event
                JOIN display_actions AS action ON action.rule_event_id = event.id
                WHERE event.analyzed_frame_id = frame.id
                    AND action.status IN ('pending', 'processing', 'retry')
            )
            AND NOT EXISTS (
                SELECT 1
                FROM rule_events AS event
                JOIN hiperwall_display_states AS state
                    ON state.active_rule_event_id = event.id
                WHERE event.analyzed_frame_id = frame.id
                    AND state.state != 'IDLE'
            )
        )
        ORDER BY frame.created_at, frame.id
        LIMIT ?
        """,
        (
            _sqlite_datetime(frame_cutoff),
            _sqlite_datetime(audit_cutoff),
            _sqlite_datetime(audit_cutoff),
            limit,
        ),
    ).fetchall()
    return tuple(int(row["id"]) for row in rows)


def _frame_counts(connection: Connection, frame_ids: tuple[int, ...]) -> RetentionCounts:
    if not frame_ids:
        return RetentionCounts()
    placeholders = ",".join("?" for _ in frame_ids)
    detections = _count(
        connection,
        f"SELECT COUNT(*) FROM detections WHERE analyzed_frame_id IN ({placeholders})",
        frame_ids,
    )
    observations = _count(
        connection,
        f"SELECT COUNT(*) FROM track_observations WHERE analyzed_frame_id IN ({placeholders})",
        frame_ids,
    )
    events = _count(
        connection,
        f"SELECT COUNT(*) FROM rule_events WHERE analyzed_frame_id IN ({placeholders})",
        frame_ids,
    )
    actions = _count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM display_actions AS action
        JOIN rule_events AS event ON event.id = action.rule_event_id
        WHERE event.analyzed_frame_id IN ({placeholders})
        """,
        frame_ids,
    )
    return RetentionCounts(
        analyzed_frames=len(frame_ids),
        detections=detections,
        track_observations=observations,
        rule_events=events,
        display_actions=actions,
    )


def _orphan_track_keys(
    connection: Connection,
    *,
    frame_cutoff: datetime,
    limit: int,
) -> tuple[tuple[str, int], ...]:
    rows = connection.execute(
        """
        SELECT track.analysis_run_id, track.track_id
        FROM tracks AS track
        WHERE track.updated_at < ?
            AND NOT EXISTS (
                SELECT 1 FROM track_observations AS observation
                WHERE observation.analysis_run_id = track.analysis_run_id
                    AND observation.track_id = track.track_id
            )
            AND NOT EXISTS (
                SELECT 1 FROM rule_events AS event
                WHERE event.analysis_run_id = track.analysis_run_id
                    AND event.track_id = track.track_id
            )
        ORDER BY track.updated_at, track.analysis_run_id, track.track_id
        LIMIT ?
        """,
        (_sqlite_datetime(frame_cutoff), limit),
    ).fetchall()
    return tuple((str(row["analysis_run_id"]), int(row["track_id"])) for row in rows)


def _preview_orphan_track_keys(
    connection: Connection,
    *,
    frame_ids: tuple[int, ...],
    frame_cutoff: datetime,
    limit: int,
) -> tuple[tuple[str, int], ...]:
    if not frame_ids:
        return _orphan_track_keys(connection, frame_cutoff=frame_cutoff, limit=limit)
    placeholders = ",".join("?" for _ in frame_ids)
    rows = connection.execute(
        f"""
        SELECT track.analysis_run_id, track.track_id
        FROM tracks AS track
        WHERE track.updated_at < ?
            AND NOT EXISTS (
                SELECT 1 FROM track_observations AS observation
                WHERE observation.analysis_run_id = track.analysis_run_id
                    AND observation.track_id = track.track_id
                    AND observation.analyzed_frame_id NOT IN ({placeholders})
            )
            AND NOT EXISTS (
                SELECT 1 FROM rule_events AS event
                WHERE event.analysis_run_id = track.analysis_run_id
                    AND event.track_id = track.track_id
                    AND event.analyzed_frame_id NOT IN ({placeholders})
            )
        ORDER BY track.updated_at, track.analysis_run_id, track.track_id
        LIMIT ?
        """,
        (_sqlite_datetime(frame_cutoff), *frame_ids, *frame_ids, limit),
    ).fetchall()
    return tuple((str(row["analysis_run_id"]), int(row["track_id"])) for row in rows)


def _count(connection: Connection, query: str, parameters: tuple[object, ...]) -> int:
    return int(connection.execute(query, parameters).fetchone()[0])


def _sqlite_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("retention cutoffs must include timezone information")
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _sqlite_utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("retention activity timestamp must include timezone information")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_limit(value: int) -> None:
    if value < 1:
        raise ValueError("retention limit must be at least 1")


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


__all__ = [
    "DatabaseActivity",
    "RetentionCounts",
    "RetentionDatabaseMetrics",
    "RetentionRepository",
    "WalCheckpointResult",
]
