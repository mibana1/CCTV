"""Container-safe SQLite inspection and verified online-backup commands."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from cctv.core.settings import get_settings
from cctv.db.database import SQLITE_BUSY_TIMEOUT_MS
from cctv.db.retention import RetentionRepository


@dataclass(frozen=True, slots=True)
class DatabaseCheckResult:
    database_path: Path
    checked_at: datetime
    integrity_ok: bool
    integrity_messages: tuple[str, ...]
    schema_version: int
    journal_mode: str
    database_bytes: int
    wal_bytes: int
    shm_bytes: int
    page_count: int
    page_size: int
    freelist_count: int


@dataclass(frozen=True, slots=True)
class DatabaseBackupResult:
    source_path: Path
    backup_path: Path
    backup_bytes: int
    completed_at: datetime
    integrity: DatabaseCheckResult


def check_database(database_path: str | Path) -> DatabaseCheckResult:
    """Run a read-only full integrity check and collect SQLite/WAL sizing metrics."""
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {path}")
    database_uri = f"{path.as_uri()}?mode=ro"
    with closing(
        sqlite3.connect(
            database_uri,
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000,
            uri=True,
        )
    ) as connection:
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA query_only = ON")
        messages = tuple(
            str(row[0]) for row in connection.execute("PRAGMA integrity_check").fetchall()
        )
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        freelist_count = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
    return DatabaseCheckResult(
        database_path=path,
        checked_at=datetime.now(UTC),
        integrity_ok=len(messages) == 1 and messages[0].casefold() == "ok",
        integrity_messages=messages,
        schema_version=schema_version,
        journal_mode=journal_mode,
        database_bytes=_size(path),
        wal_bytes=_size(Path(f"{path}-wal")),
        shm_bytes=_size(Path(f"{path}-shm")),
        page_count=page_count,
        page_size=page_size,
        freelist_count=freelist_count,
    )


def create_online_backup(
    database_path: str | Path,
    destination: str | Path,
) -> DatabaseBackupResult:
    """Create one atomic SQLite backup and retain it only after integrity_check=ok."""
    source_path = Path(database_path).expanduser().resolve()
    backup_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {source_path}")
    if source_path == backup_path:
        raise ValueError("backup destination must differ from the source database")
    completed_path = RetentionRepository(source_path).online_backup(backup_path)
    try:
        integrity = check_database(completed_path)
        if not integrity.integrity_ok:
            raise RuntimeError(
                "online backup failed SQLite integrity_check: "
                + "; ".join(integrity.integrity_messages)
            )
    except Exception:
        completed_path.unlink(missing_ok=True)
        raise
    return DatabaseBackupResult(
        source_path=source_path,
        backup_path=completed_path,
        backup_bytes=completed_path.stat().st_size,
        completed_at=datetime.now(UTC),
        integrity=integrity,
    )


def run_backup(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``cctv-db-backup`` inside the Backend container."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Create a verified SQLite online backup")
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    output = arguments.output or (
        settings.database_backup_dir / _backup_filename(datetime.now(UTC))
    )
    print(_json(create_online_backup(arguments.database, output)))


def run_check(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``cctv-db-check`` inside the Backend container."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run SQLite integrity and size checks")
    parser.add_argument("--database", type=Path, default=settings.database_path)
    arguments = parser.parse_args(argv)
    result = check_database(arguments.database)
    print(_json(result))
    if not result.integrity_ok:
        raise SystemExit(2)


def _json(value: object) -> str:
    return json.dumps(asdict(value), default=str, ensure_ascii=False, sort_keys=True)


def _backup_filename(now: datetime) -> str:
    timestamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"cctv-{timestamp}.db"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


__all__ = [
    "DatabaseBackupResult",
    "DatabaseCheckResult",
    "check_database",
    "create_online_backup",
    "run_backup",
    "run_check",
]
