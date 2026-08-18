"""SQLite connection configuration and schema initialization."""

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from cctv.db.migrations import MIGRATIONS

logger = logging.getLogger(__name__)

SQLITE_BUSY_TIMEOUT_MS = 5_000

_CREATE_MIGRATION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@dataclass(frozen=True)
class DatabaseState:
    """Result of bringing a database file to the current schema version."""

    path: Path
    schema_version: int
    applied_migrations: tuple[int, ...]


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a configured SQLite connection and create its parent directory."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(database_path: Path) -> DatabaseState:
    """Create the SQLite file and apply every pending migration once."""
    path = Path(database_path)
    newly_applied: list[int] = []

    with closing(connect_database(path)) as connection:
        connection.execute(_CREATE_MIGRATION_TABLE)
        connection.commit()

        applied_versions = {
            row["version"] for row in connection.execute("SELECT version FROM schema_migrations")
        }

        for migration in MIGRATIONS:
            if migration.version in applied_versions:
                continue

            with connection:
                migration.apply(connection)
                connection.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                    (migration.version, migration.name),
                )
                connection.execute(f"PRAGMA user_version = {migration.version}")
            newly_applied.append(migration.version)
            logger.info(
                "Database migration applied",
                extra={
                    "event": "database_migration_applied",
                    "migration_version": migration.version,
                    "migration_name": migration.name,
                },
            )

        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        schema_version = int(row["version"])
        connection.execute(f"PRAGMA user_version = {schema_version}")

    state = DatabaseState(
        path=path,
        schema_version=schema_version,
        applied_migrations=tuple(newly_applied),
    )
    logger.info(
        "Database initialized",
        extra={
            "event": "database_initialized",
            "database_path": path,
            "schema_version": schema_version,
            "applied_migrations": newly_applied,
        },
    )
    return state
