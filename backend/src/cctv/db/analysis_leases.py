"""Atomic SQLite ownership leases for camera analysis workers."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cctv.db.database import connect_database


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


__all__ = ["AnalysisLeaseRepository"]
