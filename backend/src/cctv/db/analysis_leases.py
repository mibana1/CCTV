"""Atomic SQLite ownership leases for camera analysis workers."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cctv.db.database import connect_database


@dataclass(frozen=True, slots=True)
class AnalysisLeaseRecord:
    camera_id: int
    owner_id: str
    analysis_run_id: str | None
    expires_at: datetime
    updated_at: datetime


class AnalysisLeaseRepository:
    """Allow one Backend instance to own a camera until its renewable lease expires."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def acquire(
        self,
        camera_id: int,
        owner_id: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> bool:
        if camera_id < 1:
            raise ValueError("camera_id must be at least 1")
        normalized_owner = owner_id.strip()
        if not 1 <= len(normalized_owner) <= 128:
            raise ValueError("owner_id must contain 1-128 characters")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        acquired_at = _utc(now)
        acquired_text = _utc_text(acquired_at)
        expires_text = _utc_text(acquired_at + timedelta(seconds=lease_seconds))
        with closing(connect_database(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO analysis_worker_leases (
                    camera_id, owner_id, expires_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    analysis_run_id = CASE
                        WHEN analysis_worker_leases.owner_id = excluded.owner_id
                        THEN analysis_worker_leases.analysis_run_id
                        ELSE NULL
                    END,
                    owner_id = excluded.owner_id,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                WHERE analysis_worker_leases.owner_id = excluded.owner_id
                    OR analysis_worker_leases.expires_at <= ?
                """,
                (
                    camera_id,
                    normalized_owner,
                    expires_text,
                    acquired_text,
                    acquired_text,
                ),
            )
            row = connection.execute(
                "SELECT owner_id FROM analysis_worker_leases WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
        return row is not None and row["owner_id"] == normalized_owner

    def assign_run(
        self,
        camera_id: int,
        owner_id: str,
        analysis_run_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Bind one planned run ID only while this owner has a valid camera lease."""
        assigned_at = _utc(now)
        run_id = analysis_run_id.strip()
        if not 1 <= len(run_id) <= 128:
            raise ValueError("analysis_run_id must contain 1-128 characters")
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE analysis_worker_leases
                SET analysis_run_id = ?, updated_at = ?
                WHERE camera_id = ? AND owner_id = ? AND expires_at > ?
                """,
                (
                    run_id,
                    _utc_text(assigned_at),
                    camera_id,
                    owner_id,
                    _utc_text(assigned_at),
                ),
            )
        return cursor.rowcount == 1

    def get(self, camera_id: int) -> AnalysisLeaseRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM analysis_worker_leases WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            return None
        return AnalysisLeaseRecord(
            camera_id=int(row["camera_id"]),
            owner_id=str(row["owner_id"]),
            analysis_run_id=(
                str(row["analysis_run_id"])
                if row["analysis_run_id"] is not None
                else None
            ),
            expires_at=_datetime(str(row["expires_at"])),
            updated_at=_datetime(str(row["updated_at"])),
        )

    def clear_run(self, camera_id: int, owner_id: str) -> bool:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE analysis_worker_leases
                SET analysis_run_id = NULL, updated_at = ?
                WHERE camera_id = ? AND owner_id = ?
                """,
                (_utc_text(_utc(None)), camera_id, owner_id),
            )
        return cursor.rowcount == 1

    def release(self, camera_id: int, owner_id: str) -> bool:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                DELETE FROM analysis_worker_leases
                WHERE camera_id = ? AND owner_id = ?
                """,
                (camera_id, owner_id),
            )
        return cursor.rowcount > 0

    def release_all(self, owner_id: str) -> int:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM analysis_worker_leases WHERE owner_id = ?",
                (owner_id,),
            )
        return cursor.rowcount


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("lease timestamps must include timezone information")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


__all__ = ["AnalysisLeaseRecord", "AnalysisLeaseRepository"]
