import sqlite3
from contextlib import closing
from pathlib import Path

from cctv.db import connect_database, initialize_database


def test_initialize_database_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "cctv.db"

    first = initialize_database(database_path)
    second = initialize_database(database_path)

    assert database_path.is_file()
    assert first.schema_version == 1
    assert first.applied_migrations == (1,)
    assert second.schema_version == 1
    assert second.applied_migrations == ()

    with closing(connect_database(database_path)) as connection:
        migration = connection.execute("SELECT version, name FROM schema_migrations").fetchone()
        camera_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'cameras'"
        ).fetchone()

        assert dict(migration) == {"version": 1, "name": "initial_camera_schema"}
        assert camera_table["name"] == "cameras"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000


def test_connect_database_returns_rows_by_column_name(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)

    with closing(connect_database(database_path)) as connection:
        connection.execute("INSERT INTO cameras (name) VALUES (?)", ("Test camera",))
        connection.commit()
        row = connection.execute("SELECT id, name, enabled FROM cameras").fetchone()

    assert isinstance(row, sqlite3.Row)
    assert dict(row) == {"id": 1, "name": "Test camera", "enabled": 1}
