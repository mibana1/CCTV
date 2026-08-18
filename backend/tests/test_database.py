import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import check_database_health, connect_database, initialize_database


def test_initialize_database_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "cctv.db"

    first = initialize_database(database_path)
    second = initialize_database(database_path)

    assert database_path.is_file()
    assert first.schema_version == 3
    assert first.applied_migrations == (1, 2, 3)
    assert second.schema_version == 3
    assert second.applied_migrations == ()

    with closing(connect_database(database_path)) as connection:
        migrations = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        camera_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'cameras'"
        ).fetchone()

        detection_tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table'
                    AND name IN ('analysis_runs', 'analyzed_frames', 'detections')
                """
            )
        }

        assert [dict(migration) for migration in migrations] == [
            {"version": 1, "name": "initial_camera_schema"},
            {"version": 2, "name": "detection_result_schema"},
            {"version": 3, "name": "detection_tracking_schema"},
        ]
        assert camera_table["name"] == "cameras"
        assert detection_tables == {"analysis_runs", "analyzed_frames", "detections"}
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


def test_check_database_health_reads_current_database_state(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)

    health = check_database_health(database_path)

    assert health.schema_version == 3
    assert health.journal_mode == "wal"


def test_check_database_health_does_not_create_a_missing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        check_database_health(database_path)

    assert not database_path.exists()
