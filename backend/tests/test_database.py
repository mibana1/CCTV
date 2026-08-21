import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import check_database_health, connect_database, initialize_database
from cctv.db.migrations import MIGRATIONS
from cctv.db.migrations.v0001_initial import apply as apply_v0001
from cctv.db.migrations.v0002_detection_results import apply as apply_v0002
from cctv.db.migrations.v0003_detection_tracking import apply as apply_v0003
from cctv.db.migrations.v0004_track_history import apply as apply_v0004
from cctv.db.migrations.v0005_track_active_state import apply as apply_v0005
from cctv.db.migrations.v0007_detector_type import apply as apply_v0007


def test_initialize_database_creates_schema_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "cctv.db"

    first = initialize_database(database_path)
    second = initialize_database(database_path)

    assert database_path.is_file()
    assert first.schema_version == 17
    assert first.applied_migrations == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
    )
    assert second.schema_version == 17
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
                        'tracks', 'track_observations', 'rules', 'rule_events',
                        'display_actions', 'identities', 'identity_embeddings',
                        'face_match_events', 'person_instances', 'track_identity_links'
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
            {"version": 7, "name": "detector_type"},
            {"version": 8, "name": "display_actions"},
            {"version": 9, "name": "identity_registry"},
            {"version": 10, "name": "face_match_events"},
            {"version": 11, "name": "camera_stream_paths"},
            {"version": 12, "name": "camera_rtsp_sources"},
            {"version": 13, "name": "person_instances"},
            {"version": 14, "name": "hiperwall_live_actions"},
            {"version": 15, "name": "camera_always_connected"},
            {"version": 16, "name": "candidate_free_face_events"},
            {"version": 17, "name": "rule_soft_delete"},
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
            "display_actions",
            "identities",
            "identity_embeddings",
            "face_match_events",
            "person_instances",
            "track_identity_links",
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


def test_initialize_database_upgrades_schema_version_fourteen_to_latest(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "existing.db"
    with closing(connect_database(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        for migration in MIGRATIONS[:-3]:
            migration.apply(connection)
            connection.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )
        connection.execute(
            """
            INSERT INTO cameras (name, stream_path, source_on_demand)
            VALUES ('Existing camera', 'camera', 1)
            """
        )
        connection.execute(
            """
            INSERT INTO analysis_runs (
                id, source_type, source_name, model_name, model_sha256, device,
                input_size, confidence_threshold, nms_threshold, sample_fps,
                status, started_at, completed_at
            ) VALUES ('legacy-run', 'rtsp', 'camera', 'model.onnx', ?, 'cpu',
                640, 0.25, 0.45, 2, 'completed', '2026-08-18T00:00:00Z',
                '2026-08-18T00:01:00Z')
            """,
            ("a" * 64,),
        )
        connection.execute(
            """
            INSERT INTO face_match_events (
                analysis_run_id, source_index, sample_index,
                source_timestamp_seconds, face_index, track_id,
                face_x, face_y, face_width, face_height,
                detection_confidence, match_status, rejection_reason,
                identity_id, external_id, display_name,
                best_candidate_identity_id, best_candidate_external_id,
                best_similarity, second_best_similarity,
                similarity_threshold, minimum_margin
            ) VALUES (
                'legacy-run', 1, 1, 0.5, 0, NULL,
                10, 20, 30, 40, 0.9, 'unknown', 'below_threshold',
                NULL, NULL, NULL, 'legacy-candidate', NULL,
                0.2, NULL, 0.45, 0.05
            )
            """
        )
        connection.execute("PRAGMA user_version = 14")

    state = initialize_database(database_path)

    assert state.schema_version == 17
    assert state.applied_migrations == (15, 16, 17)
    with closing(connect_database(database_path)) as connection:
        camera = connection.execute(
            """
            SELECT name, stream_path, rtsp_source_ciphertext, source_on_demand,
                provisioning_status
            FROM cameras
            """
        ).fetchone()
        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table'
                    AND name IN (
                        'identities', 'identity_embeddings', 'face_match_events',
                        'person_instances', 'track_identity_links'
                    )
                """
            )
        }
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        legacy_face_event = connection.execute(
            """
            SELECT rejection_reason, best_candidate_identity_id, best_similarity
            FROM face_match_events WHERE analysis_run_id = 'legacy-run'
            """
        ).fetchone()

    assert dict(camera) == {
        "name": "Existing camera",
        "stream_path": "camera",
        "rtsp_source_ciphertext": None,
        "source_on_demand": 0,
        "provisioning_status": "external",
    }
    assert tables == {
        "identities",
        "identity_embeddings",
        "face_match_events",
        "person_instances",
        "track_identity_links",
    }
    assert foreign_key_errors == []
    assert dict(legacy_face_event) == {
        "rejection_reason": "below_threshold",
        "best_candidate_identity_id": "legacy-candidate",
        "best_similarity": 0.2,
    }


def test_check_database_health_reads_current_database_state(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)

    health = check_database_health(database_path)

    assert health.schema_version == 17
    assert health.journal_mode == "wal"


def test_check_database_health_does_not_create_a_missing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "missing.db"

    with pytest.raises(sqlite3.OperationalError):
        check_database_health(database_path)

    assert not database_path.exists()


def test_detector_type_migration_backfills_historical_runs(tmp_path: Path) -> None:
    database_path = tmp_path / "historical.db"
    with closing(connect_database(database_path)) as connection, connection:
        apply_v0001(connection)
        apply_v0002(connection)
        connection.execute(
            """
            INSERT INTO analysis_runs (
                id, source_type, source_name, model_name, model_sha256, device,
                input_size, confidence_threshold, nms_threshold, sample_fps,
                status, started_at
            ) VALUES ('old-run', 'local_video', 'sample.mp4', 'model.onnx', ?,
                'cpu', 640, 0.25, 0.45, 2, 'running', '2026-08-18T00:00:00Z')
            """,
            ("a" * 64,),
        )

        apply_v0007(connection)
        detector_type = connection.execute(
            "SELECT detector_type FROM analysis_runs WHERE id = 'old-run'"
        ).fetchone()[0]

    assert detector_type == "yolo_onnx"


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
