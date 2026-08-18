import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import check_database_health, connect_database, initialize_database
from cctv.db.migrations.v0001_initial import apply as apply_v0001
from cctv.db.migrations.v0002_detection_results import apply as apply_v0002
from cctv.db.migrations.v0003_detection_tracking import apply as apply_v0003
from cctv.db.migrations.v0004_track_history import apply as apply_v0004
from cctv.db.migrations.v0005_track_active_state import apply as apply_v0005


def test_initialize_database_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "cctv.db"

    first = initialize_database(database_path)
    second = initialize_database(database_path)

    assert database_path.is_file()
    assert first.schema_version == 6
    assert first.applied_migrations == (1, 2, 3, 4, 5, 6)
    assert second.schema_version == 6
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
                    AND name IN (
                        'analysis_runs', 'analyzed_frames', 'detections',
                        'tracks', 'track_observations', 'rules', 'rule_events'
                    )
                """
            )
        }

        assert [dict(migration) for migration in migrations] == [
            {"version": 1, "name": "initial_camera_schema"},
            {"version": 2, "name": "detection_result_schema"},
            {"version": 3, "name": "detection_tracking_schema"},
            {"version": 4, "name": "track_history_schema"},
            {"version": 5, "name": "track_active_state_schema"},
            {"version": 6, "name": "rule_engine_schema"},
        ]
        assert camera_table["name"] == "cameras"
        assert detection_tables == {
            "analysis_runs",
            "analyzed_frames",
            "detections",
            "tracks",
            "track_observations",
            "rules",
            "rule_events",
        }
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

    assert health.schema_version == 6
    assert health.journal_mode == "wal"


def test_check_database_health_does_not_create_a_missing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        check_database_health(database_path)

    assert not database_path.exists()


def test_track_history_migration_backfills_existing_tracked_detections(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    with closing(connect_database(database_path)) as connection, connection:
        apply_v0001(connection)
        apply_v0002(connection)
        apply_v0003(connection)
        connection.execute(
            """
            INSERT INTO analysis_runs (
                id, source_type, source_name, model_name, model_sha256, device,
                input_size, confidence_threshold, nms_threshold, sample_fps,
                status, started_at
            ) VALUES (?, 'local_video', 'sample.mp4', 'model.onnx', ?, 'cpu',
                640, 0.25, 0.45, 2, 'running', '2026-08-18T00:00:00Z')
            """,
            ("historical-run", "a" * 64),
        )
        frame_id = connection.execute(
            """
            INSERT INTO analyzed_frames (
                analysis_run_id, source_index, sample_index,
                source_timestamp_seconds, frame_width, frame_height,
                inference_seconds, detection_count
            ) VALUES ('historical-run', 10, 5, 2.5, 640, 360, 0.02, 1)
            """
        ).lastrowid
        connection.execute(
            """
            INSERT INTO detections (
                analyzed_frame_id, track_id, class_id, class_name, confidence,
                x1, y1, x2, y2
            ) VALUES (?, 7, 0, 'person', 0.91, 10, 20, 100, 200)
            """,
            (frame_id,),
        )

        apply_v0004(connection)
        apply_v0004(connection)
        apply_v0005(connection)
        apply_v0005(connection)

        track = connection.execute("SELECT * FROM tracks").fetchone()
        observation = connection.execute("SELECT * FROM track_observations").fetchone()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert track["analysis_run_id"] == "historical-run"
    assert track["track_id"] == 7
    assert track["class_name"] == "person"
    assert track["first_sample_index"] == 5
    assert track["last_sample_index"] == 5
    assert track["observation_count"] == 1
    assert track["max_confidence"] == pytest.approx(0.91)
    assert track["is_active"] == 0
    assert observation["analysis_run_id"] == "historical-run"
    assert observation["track_id"] == 7
    assert foreign_key_errors == []
