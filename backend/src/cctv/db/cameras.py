"""SQLite repository for dashboard camera registrations."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from cctv.db.database import connect_database
from cctv.db.detections import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

MAX_CAMERA_NAME_LENGTH = 100
MAX_CAMERA_LOCATION_LENGTH = 200
MAX_STREAM_PATH_LENGTH = 128
_STREAM_PATH_PATTERN = re.compile(r"[A-Za-z0-9_.~-]+(?:/[A-Za-z0-9_.~-]+)*")


class CameraProvisioningStatus(StrEnum):
    """Current relationship between a registration and MediaMTX."""

    EXTERNAL = "external"
    PENDING = "pending"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CameraRecord:
    """One logical camera mapped to a MediaMTX stream path."""

    id: int
    name: str
    location: str | None
    stream_path: str
    enabled: bool
    rtsp_endpoint: str | None
    credentials_configured: bool
    source_on_demand: bool
    provisioning_status: CameraProvisioningStatus
    last_sync_error: str | None
    last_synced_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CameraPage:
    items: tuple[CameraRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class CameraSourceRecord:
    """Internal encrypted source material used only by the provisioning service."""

    camera_id: int
    stream_path: str
    enabled: bool
    source_on_demand: bool
    ciphertext: str | None


class CameraRepository:
    """Create and manage camera metadata without storing RTSP credentials."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def create_camera(
        self,
        *,
        name: str,
        stream_path: str,
        location: str | None = None,
        enabled: bool = True,
        rtsp_endpoint: str | None = None,
        rtsp_source_ciphertext: str | None = None,
        source_on_demand: bool = True,
        provisioning_status: CameraProvisioningStatus = CameraProvisioningStatus.EXTERNAL,
    ) -> CameraRecord:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO cameras (
                    name, location, stream_path, enabled, rtsp_endpoint,
                    rtsp_source_ciphertext, source_on_demand, provisioning_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _camera_name(name),
                    _optional_location(location),
                    validate_stream_path(stream_path),
                    int(_enabled(enabled)),
                    _optional_endpoint(rtsp_endpoint),
                    _optional_ciphertext(rtsp_source_ciphertext),
                    int(_enabled(source_on_demand)),
                    CameraProvisioningStatus(provisioning_status),
                ),
            )
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        if row is None:
            raise RuntimeError("camera was not created")
        return _camera_from_row(row)

    def get_camera(self, camera_id: int) -> CameraRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?",
                (camera_id,),
            ).fetchone()
        return _camera_from_row(row) if row is not None else None

    def get_camera_source(self, camera_id: int) -> CameraSourceRecord | None:
        """Load encrypted source data without exposing it through a CameraRecord."""
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT id, stream_path, enabled, source_on_demand, rtsp_source_ciphertext
                FROM cameras
                WHERE id = ?
                """,
                (camera_id,),
            ).fetchone()
        return _camera_source_from_row(row) if row is not None else None

    def list_managed_camera_sources(self) -> tuple[CameraSourceRecord, ...]:
        """Load every dashboard-managed camera for reconciliation."""
        with closing(connect_database(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT id, stream_path, enabled, source_on_demand, rtsp_source_ciphertext
                FROM cameras
                WHERE rtsp_source_ciphertext IS NOT NULL
                ORDER BY id
                """
            ).fetchall()
        return tuple(_camera_source_from_row(row) for row in rows)

    def list_cameras(
        self,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        enabled: bool | None = None,
    ) -> CameraPage:
        if page < 1:
            raise ValueError("page must be at least 1")
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        where_sql = " WHERE enabled = ?" if enabled is not None else ""
        parameters: tuple[object, ...] = (int(enabled),) if enabled is not None else ()
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM cameras{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM cameras{where_sql}
                ORDER BY enabled DESC, created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return CameraPage(
            items=tuple(_camera_from_row(row) for row in rows),
            total=total,
        )

    def update_camera(self, camera_id: int, changes: Mapping[str, Any]) -> CameraRecord:
        allowed = {"name", "location", "stream_path", "enabled"}
        unexpected = set(changes) - allowed
        if unexpected:
            raise ValueError(f"unsupported camera fields: {', '.join(sorted(unexpected))}")
        if not changes:
            raise ValueError("at least one camera field is required")

        assignments: list[str] = []
        parameters: list[object] = []
        for field, value in changes.items():
            assignments.append(f"{field} = ?")
            if field == "name":
                parameters.append(_camera_name(value))
            elif field == "location":
                parameters.append(_optional_location(value))
            elif field == "stream_path":
                parameters.append(validate_stream_path(value))
            elif field == "enabled":
                parameters.append(int(_enabled(value)))
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        parameters.append(camera_id)

        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                f"UPDATE cameras SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise LookupError(f"camera does not exist: {camera_id}")
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("updated camera could not be loaded")
        return _camera_from_row(row)

    def update_camera_source(
        self,
        camera_id: int,
        *,
        rtsp_endpoint: str,
        rtsp_source_ciphertext: str,
        source_on_demand: bool,
    ) -> CameraRecord:
        """Replace encrypted connection data and mark it for synchronization."""
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE cameras
                SET rtsp_endpoint = ?, rtsp_source_ciphertext = ?, source_on_demand = ?,
                    provisioning_status = 'pending', last_sync_error = NULL,
                    last_synced_at = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    _optional_endpoint(rtsp_endpoint),
                    _optional_ciphertext(rtsp_source_ciphertext),
                    int(_enabled(source_on_demand)),
                    camera_id,
                ),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"camera does not exist: {camera_id}")
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("updated camera source could not be loaded")
        return _camera_from_row(row)

    def set_provisioning_result(
        self,
        camera_id: int,
        *,
        provisioning_status: CameraProvisioningStatus,
        error: str | None = None,
    ) -> CameraRecord:
        """Persist a sanitized synchronization result."""
        normalized_error = _optional_sync_error(error)
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE cameras
                SET provisioning_status = ?, last_sync_error = ?,
                    last_synced_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (CameraProvisioningStatus(provisioning_status), normalized_error, camera_id),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"camera does not exist: {camera_id}")
            row = connection.execute(
                "SELECT * FROM cameras WHERE id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("camera provisioning result could not be loaded")
        return _camera_from_row(row)

    def delete_camera(self, camera_id: int) -> bool:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        return cursor.rowcount > 0


def validate_stream_path(value: object) -> str:
    """Validate a relative MediaMTX path that is safe to append to an HTTP origin."""
    if not isinstance(value, str):
        raise TypeError("stream_path must be a string")
    normalized = value.strip().strip("/")
    if (
        not normalized
        or len(normalized) > MAX_STREAM_PATH_LENGTH
        or not _STREAM_PATH_PATTERN.fullmatch(normalized)
        or ".." in normalized.split("/")
    ):
        raise ValueError(
            "stream_path must be 1-128 URL-safe path characters without empty or '..' segments"
        )
    return normalized


def _camera_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("name must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > MAX_CAMERA_NAME_LENGTH:
        raise ValueError(f"name must contain 1-{MAX_CAMERA_NAME_LENGTH} characters")
    return normalized


def _optional_location(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("location must be a string or null")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > MAX_CAMERA_LOCATION_LENGTH:
        raise ValueError(f"location must contain at most {MAX_CAMERA_LOCATION_LENGTH} characters")
    return normalized


def _enabled(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("enabled must be a boolean")
    return value


def _optional_endpoint(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("rtsp_endpoint must be a string or null")
    normalized = value.strip()
    if not normalized or len(normalized) > 2_048:
        raise ValueError("rtsp_endpoint must contain 1-2048 characters")
    return normalized


def _optional_ciphertext(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("rtsp_source_ciphertext must be a string or null")
    normalized = value.strip()
    if not normalized or len(normalized) > 8_192:
        raise ValueError("rtsp_source_ciphertext must contain 1-8192 characters")
    return normalized


def _optional_sync_error(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("error must be a string or null")
    normalized = value.strip()
    return normalized[:500] or None


def _camera_from_row(row: sqlite3.Row) -> CameraRecord:
    return CameraRecord(
        id=int(row["id"]),
        name=str(row["name"]),
        location=str(row["location"]) if row["location"] is not None else None,
        stream_path=validate_stream_path(row["stream_path"]),
        enabled=bool(row["enabled"]),
        rtsp_endpoint=(str(row["rtsp_endpoint"]) if row["rtsp_endpoint"] is not None else None),
        credentials_configured=row["rtsp_source_ciphertext"] is not None,
        source_on_demand=bool(row["source_on_demand"]),
        provisioning_status=CameraProvisioningStatus(str(row["provisioning_status"])),
        last_sync_error=(
            str(row["last_sync_error"]) if row["last_sync_error"] is not None else None
        ),
        last_synced_at=(
            _parse_datetime(row["last_synced_at"])
            if row["last_synced_at"] is not None
            else None
        ),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _camera_source_from_row(row: sqlite3.Row) -> CameraSourceRecord:
    return CameraSourceRecord(
        camera_id=int(row["id"]),
        stream_path=validate_stream_path(row["stream_path"]),
        enabled=bool(row["enabled"]),
        source_on_demand=bool(row["source_on_demand"]),
        ciphertext=(
            str(row["rtsp_source_ciphertext"])
            if row["rtsp_source_ciphertext"] is not None
            else None
        ),
    )


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


__all__ = [
    "MAX_CAMERA_LOCATION_LENGTH",
    "MAX_CAMERA_NAME_LENGTH",
    "MAX_STREAM_PATH_LENGTH",
    "CameraPage",
    "CameraProvisioningStatus",
    "CameraRecord",
    "CameraRepository",
    "CameraSourceRecord",
    "validate_stream_path",
]
