"""Atomic SQLite ownership leases for singleton maintenance jobs."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cctv.db.database import connect_database


class MaintenanceLeaseRepository:
    """Allow only one Backend instance to own a named maintenance job."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def acquire(
        self,
        name: str,
        owner_id: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> bool:
        normalized_name = name.strip()
        normalized_owner = owner_id.strip()
        if not 1 <= len(normalized_name) <= 64:
            raise ValueError("lease name must contain 1-64 characters")
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
                INSERT INTO maintenance_leases (name, owner_id, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                WHERE maintenance_leases.owner_id = excluded.owner_id
                    OR maintenance_leases.expires_at <= ?
                """,
                (
                    normalized_name,
                    normalized_owner,
                    expires_text,
                    acquired_text,
                    acquired_text,
                ),
            )
            row = connection.execute(
                "SELECT owner_id FROM maintenance_leases WHERE name = ?",
                (normalized_name,),
            ).fetchone()
        return row is not None and row["owner_id"] == normalized_owner

    def release(self, name: str, owner_id: str) -> bool:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM maintenance_leases WHERE name = ? AND owner_id = ?",
                (name, owner_id),
            )
        return cursor.rowcount > 0


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("lease timestamps must include timezone information")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = ["MaintenanceLeaseRepository"]
