from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cctv.db import (
    AnalysisLeaseRepository,
    AnalysisRunRecoveryRepository,
    AnalysisRunStatus,
    AnalysisWorkerFailureRepository,
    CameraRepository,
    DetectionRepository,
    RuleRepository,
    connect_database,
    initialize_database,
)

NOW = datetime(2026, 8, 21, 7, 30, tzinfo=UTC)


def _create_run(
    repository: DetectionRepository,
    *,
    camera_id: int,
    source_name: str,
    run_id: str,
) -> None:
    repository.create_analysis_run(
        source_type="rtsp",
        source_name=source_name,
        camera_id=camera_id,
        detector_type="test",
        model_name="model.onnx",
        model_sha256="a" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
        run_id=run_id,
    )


def _insert_active_track(database_path: Path, run_id: str, track_id: int) -> None:
    with closing(connect_database(database_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO tracks (
                analysis_run_id, track_id, class_id, class_name,
                first_sample_index, last_sample_index,
                first_seen_timestamp_seconds, last_seen_timestamp_seconds,
                observation_count, max_confidence, is_active
            ) VALUES (?, ?, 0, 'person', 0, 0, 0, 0, 1, 0.9, 1)
            """,
            (run_id, track_id),
        )


def test_startup_recovery_protects_valid_leases_and_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    cameras = CameraRepository(database_path)
    runs = DetectionRepository(database_path)
    leases = AnalysisLeaseRepository(database_path)
    recovery = AnalysisRunRecoveryRepository(database_path)
    valid_camera = cameras.create_camera(name="Valid", stream_path="valid")
    expired_camera = cameras.create_camera(name="Expired", stream_path="expired")
    orphan_camera = cameras.create_camera(name="Orphan", stream_path="orphan")
    new_camera = cameras.create_camera(name="New", stream_path="new")
    for camera, run_id in (
        (valid_camera, "valid-run"),
        (expired_camera, "expired-run"),
        (orphan_camera, "orphan-run"),
        (new_camera, "new-run"),
    ):
        _create_run(
            runs,
            camera_id=camera.id,
            source_name=camera.stream_path,
            run_id=run_id,
        )
        _insert_active_track(database_path, run_id, camera.id)

    assert leases.acquire(valid_camera.id, "backend-a", lease_seconds=60, now=NOW)
    assert leases.assign_run(valid_camera.id, "backend-a", "valid-run", now=NOW)
    expired_at = NOW - timedelta(seconds=61)
    assert leases.acquire(
        expired_camera.id,
        "backend-old",
        lease_seconds=30,
        now=expired_at,
    )
    assert leases.assign_run(
        expired_camera.id,
        "backend-old",
        "expired-run",
        now=expired_at,
    )

    result = recovery.recover_orphaned_runs(now=NOW, excluded_run_ids=("new-run",))
    repeated = recovery.recover_orphaned_runs(now=NOW + timedelta(seconds=1))

    assert result.recovered_run_ids == ("expired-run", "orphan-run")
    assert result.closed_active_track_count == 2
    assert repeated.recovered_run_count == 1
    assert repeated.recovered_run_ids == ("new-run",)
    assert recovery.recover_orphaned_runs(
        now=NOW + timedelta(seconds=2)
    ).recovered_run_count == 0
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT id, status, completed_at, completion_reason
            FROM analysis_runs ORDER BY id
            """
        ).fetchall()
        tracks = connection.execute(
            "SELECT analysis_run_id, is_active FROM tracks ORDER BY analysis_run_id"
        ).fetchall()

    by_id = {str(row["id"]): row for row in rows}
    assert by_id["valid-run"]["status"] == AnalysisRunStatus.RUNNING
    assert by_id["valid-run"]["completed_at"] is None
    for run_id in ("expired-run", "orphan-run", "new-run"):
        assert by_id[run_id]["status"] == AnalysisRunStatus.INTERRUPTED
        assert by_id[run_id]["completed_at"] is not None
        assert by_id[run_id]["completion_reason"] == "recovered_on_startup"
    assert {str(row["analysis_run_id"]): row["is_active"] for row in tracks} == {
        "expired-run": 0,
        "new-run": 0,
        "orphan-run": 0,
        "valid-run": 1,
    }


def test_startup_recovery_preserves_hiperwall_gate_and_person_history(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    camera = CameraRepository(database_path).create_camera(name="Lobby", stream_path="lobby")
    rule = RuleRepository(database_path).create_rule(
        name="Lobby rule",
        source_name="lobby",
        rule_type="intrusion",
        class_name="person",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
    )
    _create_run(
        DetectionRepository(database_path),
        camera_id=camera.id,
        source_name="lobby",
        run_id="orphan-run",
    )
    _insert_active_track(database_path, "orphan-run", 1)
    with closing(connect_database(database_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO person_instances (
                id, analysis_run_id, status, identity_id, external_id, display_name,
                best_candidate_identity_id, best_candidate_external_id, best_similarity,
                first_sample_index, last_sample_index,
                first_seen_timestamp_seconds, last_seen_timestamp_seconds,
                last_face_x, last_face_y, last_face_width, last_face_height, track_count
            ) VALUES (
                'person-1', 'orphan-run', 'unknown', NULL, NULL, NULL,
                'candidate-1', NULL, 0.2, 0, 0, 0, 0, 1, 2, 10, 10, 1
            )
            """
        )
        connection.execute(
            """
            INSERT INTO hiperwall_display_states (
                rule_id, source_name, state, active_rule_event_id, open_action_id,
                instance_id, cooldown_seconds, displaying_since, created_at, updated_at
            ) VALUES (?, 'lobby', 'DISPLAYING', 'event-1', 'open-1',
                'instance-1', 30, ?, ?, ?)
            """,
            (rule.id, NOW.isoformat(), NOW.isoformat(), NOW.isoformat()),
        )

    result = AnalysisRunRecoveryRepository(database_path).recover_orphaned_runs(now=NOW)

    assert result.preserved_person_instance_count == 1
    with closing(connect_database(database_path)) as connection:
        person_count = connection.execute(
            "SELECT COUNT(*) FROM person_instances WHERE id = 'person-1'"
        ).fetchone()[0]
        gate = connection.execute(
            """
            SELECT state, active_rule_event_id, open_action_id, instance_id
            FROM hiperwall_display_states WHERE rule_id = ? AND source_name = 'lobby'
            """,
            (rule.id,),
        ).fetchone()
    assert person_count == 1
    assert dict(gate) == {
        "state": "DISPLAYING",
        "active_rule_event_id": "event-1",
        "open_action_id": "open-1",
        "instance_id": "instance-1",
    }


def test_worker_failure_history_is_append_only_idempotent_and_sanitized(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    camera = CameraRepository(database_path).create_camera(name="Lobby", stream_path="lobby")
    failures = AnalysisWorkerFailureRepository(database_path)
    first = failures.record(
        camera_id=camera.id,
        analysis_run_id=None,
        owner_id="backend-a",
        process_instance_id="process-1",
        failure_type="RtspAuthError",
        failure_message=(
            "open rtsp://admin:camera-pass@camera.local/live "
            "failed token=secret-token"
        ),
        failure_at=NOW,
        restart_count=1,
    )
    duplicate = failures.record(
        camera_id=camera.id,
        analysis_run_id=None,
        owner_id="backend-a",
        process_instance_id="process-1",
        failure_type="Ignored",
        failure_message="ignored",
        failure_at=NOW + timedelta(seconds=1),
        restart_count=2,
    )

    assert duplicate.id == first.id
    assert "camera-pass" not in first.failure_message
    assert "secret-token" not in first.failure_message
    assert "***" in first.failure_message
    assert failures.latest(camera.id) == first
    assert failures.list_since(camera.id, since=NOW - timedelta(seconds=1)) == (first,)
