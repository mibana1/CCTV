import logging
import os
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import (
    DatabaseActivity,
    DetectionRepository,
    IdentityRepository,
    MaintenanceLeaseRepository,
    RetentionCounts,
    RetentionDatabaseMetrics,
    RetentionRepository,
    RuleRepository,
    WalCheckpointResult,
    connect_database,
    initialize_database,
)
from cctv.hiperwall import DisplayAction
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.main import create_app
from cctv.rules import RuleEvent
from cctv.workers import RetentionRunStatus, RetentionWorker, VacuumRunStatus
from cctv.workers.retention import build_argument_parser

_NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "app_env": "test",
        "database_path": tmp_path / "cctv.db",
        "model_path": tmp_path / "model.onnx",
        "log_path": tmp_path / "cctv.jsonl",
        "snapshot_dir": tmp_path / "snapshots",
        "identity_photo_dir": tmp_path / "snapshots" / "registered-identities",
        "retention_frame_days": 7,
        "retention_audit_days": 90,
        "retention_snapshot_days": 30,
        "retention_batch_size": 1,
        "retention_max_batches_per_run": 10,
    }
    values.update(overrides)
    return Settings(**values)


def _frame(sample_index: int, track_id: int) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index,
        sample_index=sample_index,
        timestamp_seconds=float(sample_index),
        frame_width=640,
        frame_height=360,
        inference_seconds=0.01,
        detections=(
            Detection(
                0,
                "person",
                0.9,
                BoundingBox(10, 20, 110, 220),
                track_id=track_id,
            ),
        ),
    )


def _event(rule_id: str, event_id: str, track_id: int) -> RuleEvent:
    return RuleEvent(
        id=event_id,
        rule_id=rule_id,
        track_id=track_id,
        event_type="intrusion_occurred",
        event_state="occurred",
        occurred_at_seconds=float(track_id),
        class_name="person",
        confidence=0.9,
    )


def _action(event_id: str, action_id: str) -> DisplayAction:
    return DisplayAction(
        id=action_id,
        rule_event_id=event_id,
        action_type="open_source",
        request={"instance_id": action_id},
        result={"external_request_sent": False},
    )


def _set_created_at(
    database_path: Path,
    *,
    frame_id: int,
    track_id: int,
    created_at: datetime,
    event_id: str | None = None,
    action_id: str | None = None,
) -> None:
    timestamp = created_at.strftime("%Y-%m-%d %H:%M:%S")
    with closing(connect_database(database_path)) as connection, connection:
        connection.execute(
            "UPDATE analyzed_frames SET created_at = ? WHERE id = ?",
            (timestamp, frame_id),
        )
        connection.execute(
            "UPDATE tracks SET created_at = ?, updated_at = ? WHERE track_id = ?",
            (timestamp, timestamp, track_id),
        )
        if event_id is not None:
            connection.execute(
                "UPDATE rule_events SET created_at = ? WHERE id = ?",
                (timestamp, event_id),
            )
        if action_id is not None:
            connection.execute(
                "UPDATE display_actions SET created_at = ?, updated_at = ? WHERE id = ?",
                (timestamp, timestamp, action_id),
            )


def _seed_retention_data(settings: Settings) -> tuple[IdentityRepository, str]:
    initialize_database(settings.database_path)
    detections = DetectionRepository(settings.database_path)
    run = detections.create_analysis_run(
        source_type="rtsp",
        source_name="camera-1",
        model_name="model.onnx",
        model_sha256="a" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
    )
    rule = RuleRepository(settings.database_path).create_rule(
        rule_id="rule-1",
        name="Restricted area",
        source_name="camera-1",
        rule_type="intrusion",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
    )

    old_frame = detections.save_frame(run.id, _frame(0, 1))
    recent_event = _event(rule.id, "event-recent", 2)
    recent_action = _action(recent_event.id, "action-recent")
    recent_event_frame = detections.save_frame(
        run.id,
        _frame(1, 2),
        rule_events=(recent_event,),
        display_actions=(recent_action,),
    )
    expired_event = _event(rule.id, "event-expired", 3)
    expired_action = _action(expired_event.id, "action-expired")
    expired_event_frame = detections.save_frame(
        run.id,
        _frame(2, 3),
        rule_events=(expired_event,),
        display_actions=(expired_action,),
    )
    pending_event = _event(rule.id, "event-pending", 4)
    pending_action = _action(pending_event.id, "action-pending")
    pending_event_frame = detections.save_frame(
        run.id,
        _frame(3, 4),
        rule_events=(pending_event,),
        display_actions=(pending_action,),
    )
    fresh_frame = detections.save_frame(run.id, _frame(4, 5))
    recent_action_event = _event(rule.id, "event-recent-action", 6)
    recent_action = _action(recent_action_event.id, "action-recent-action")
    recent_action_frame = detections.save_frame(
        run.id,
        _frame(5, 6),
        rule_events=(recent_action_event,),
        display_actions=(recent_action,),
    )

    _set_created_at(
        settings.database_path,
        frame_id=old_frame,
        track_id=1,
        created_at=_NOW - timedelta(days=20),
    )
    _set_created_at(
        settings.database_path,
        frame_id=recent_event_frame,
        track_id=2,
        event_id=recent_event.id,
        action_id=recent_action.id,
        created_at=_NOW - timedelta(days=30),
    )
    _set_created_at(
        settings.database_path,
        frame_id=expired_event_frame,
        track_id=3,
        event_id=expired_event.id,
        action_id=expired_action.id,
        created_at=_NOW - timedelta(days=120),
    )
    _set_created_at(
        settings.database_path,
        frame_id=pending_event_frame,
        track_id=4,
        event_id=pending_event.id,
        action_id=pending_action.id,
        created_at=_NOW - timedelta(days=120),
    )
    _set_created_at(
        settings.database_path,
        frame_id=fresh_frame,
        track_id=5,
        created_at=_NOW - timedelta(days=1),
    )
    _set_created_at(
        settings.database_path,
        frame_id=recent_action_frame,
        track_id=6,
        event_id=recent_action_event.id,
        action_id=recent_action.id,
        created_at=_NOW - timedelta(days=120),
    )
    with closing(connect_database(settings.database_path)) as connection, connection:
        connection.execute(
            "UPDATE display_actions SET mode = 'live', status = 'pending' WHERE id = ?",
            (pending_action.id,),
        )
        connection.execute(
            "UPDATE display_actions SET created_at = ?, updated_at = ? WHERE id = ?",
            (
                (_NOW - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
                (_NOW - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),
                recent_action.id,
            ),
        )

    identities = IdentityRepository(settings.database_path)
    identity = identities.create_identity(display_name="Registered person")
    identities.add_embedding(identity.id, model_name="test", vector=(0.1, 0.2))
    return identities, identity.id


def _write_snapshot(path: Path, *, age_days: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"snapshot")
    timestamp = (_NOW - timedelta(days=age_days)).timestamp()
    os.utime(path, (timestamp, timestamp))


def test_retention_dry_run_and_apply_preserve_audit_and_identity_data(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    identities, identity_id = _seed_retention_data(settings)
    old_snapshot = settings.snapshot_dir / "old.jpg"
    fresh_snapshot = settings.snapshot_dir / "fresh.jpg"
    identity_photo = settings.identity_photo_dir / identity_id / "registered.jpg"
    _write_snapshot(old_snapshot, age_days=40)
    _write_snapshot(fresh_snapshot, age_days=1)
    _write_snapshot(identity_photo, age_days=365)
    worker = RetentionWorker(settings, owner_id="backend-a", now_factory=lambda: _NOW)

    preview = worker.run_once(dry_run=True)

    assert preview.status is RetentionRunStatus.COMPLETED
    assert preview.counts.analyzed_frames == 2
    assert preview.counts.detections == 2
    assert preview.counts.track_observations == 2
    assert preview.counts.tracks == 2
    assert preview.counts.rule_events == 1
    assert preview.counts.display_actions == 1
    assert preview.counts.snapshots == 1
    assert preview.checkpoint is None
    assert preview.vacuum is None
    assert old_snapshot.exists()

    result = worker.run_once(dry_run=False)

    assert result.status is RetentionRunStatus.COMPLETED
    assert result.database_batches == 2
    assert result.snapshot_batches == 1
    assert result.counts == preview.counts
    assert result.database_before is not None
    assert result.database_after is not None
    assert result.checkpoint is not None
    assert result.checkpoint.mode == "passive"
    assert result.vacuum is not None
    assert result.vacuum.status is VacuumRunStatus.DISABLED
    assert not old_snapshot.exists()
    assert fresh_snapshot.exists()
    assert identity_photo.exists()
    assert identities.get_identity(identity_id) is not None
    assert identities.list_embeddings(identity_id).total == 1

    with closing(connect_database(settings.database_path)) as connection:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "analyzed_frames",
                "detections",
                "track_observations",
                "tracks",
                "rule_events",
                "display_actions",
            )
        }
        remaining_events = {
            str(row[0]) for row in connection.execute("SELECT id FROM rule_events").fetchall()
        }
    assert counts == {
        "analyzed_frames": 4,
        "detections": 4,
        "track_observations": 4,
        "tracks": 4,
        "rule_events": 3,
        "display_actions": 3,
    }
    assert remaining_events == {
        "event-recent",
        "event-pending",
        "event-recent-action",
    }


def test_retention_lease_prevents_parallel_backend_cleanup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    leases = MaintenanceLeaseRepository(settings.database_path)
    assert leases.acquire("data_retention", "backend-a", lease_seconds=300, now=_NOW)
    worker = RetentionWorker(settings, owner_id="backend-b", now_factory=lambda: _NOW)

    result = worker.run_once(dry_run=False)

    assert result.status is RetentionRunStatus.SKIPPED
    assert result.lease_acquired is False


def test_retention_preserves_event_used_by_current_hiperwall_display(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_retention_data(settings)
    timestamp = (_NOW - timedelta(days=120)).strftime("%Y-%m-%d %H:%M:%S")
    with closing(connect_database(settings.database_path)) as connection, connection:
        connection.execute(
            "UPDATE display_actions SET status = 'succeeded' WHERE id = 'action-pending'"
        )
        connection.execute(
            """
            INSERT INTO hiperwall_display_states (
                rule_id, source_name, state, active_rule_event_id, open_action_id,
                instance_id, cooldown_seconds, displaying_since, created_at, updated_at
            ) VALUES (
                'rule-1', 'camera-1', 'DISPLAYING', 'event-pending', 'action-pending',
                'instance-pending', 0, ?, ?, ?
            )
            """,
            (timestamp, timestamp, timestamp),
        )

    RetentionWorker(settings, now_factory=lambda: _NOW).run_once(dry_run=False)

    with closing(connect_database(settings.database_path)) as connection:
        event = connection.execute(
            "SELECT id FROM rule_events WHERE id = 'event-pending'"
        ).fetchone()
        action = connection.execute(
            "SELECT id FROM display_actions WHERE id = 'action-pending'"
        ).fetchone()
    assert event is not None
    assert action is not None


def test_retention_logs_failure_reason_and_database_metrics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    metrics = RetentionDatabaseMetrics(
        database_bytes=100,
        wal_bytes=20,
        shm_bytes=10,
        page_count=5,
        page_size=4_096,
        freelist_count=2,
    )

    class FailingRepository:
        def metrics(self) -> RetentionDatabaseMetrics:
            return metrics

        def delete_batch(self, **_: object) -> None:
            raise RuntimeError("simulated retention failure")

    worker = RetentionWorker(
        settings,
        repository=FailingRepository(),  # type: ignore[arg-type]
        now_factory=lambda: _NOW,
    )

    with (
        caplog.at_level(logging.ERROR, logger="cctv.workers.retention"),
        pytest.raises(RuntimeError, match="simulated retention failure"),
    ):
        worker.run_once(dry_run=False)

    failure = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "retention_cleanup_failed"
    )
    assert failure.error_type == "RuntimeError"  # type: ignore[attr-defined]
    assert failure.database_before == {  # type: ignore[attr-defined]
        "database_bytes": 100,
        "wal_bytes": 20,
        "shm_bytes": 10,
        "page_count": 5,
        "page_size": 4_096,
        "freelist_count": 2,
    }


def test_retention_repository_runs_checkpoint_and_verified_online_backup(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    repository = RetentionRepository(settings.database_path)

    checkpoint = repository.checkpoint(mode="passive")
    backup_path = repository.online_backup(tmp_path / "backups" / "pre-vacuum.sqlite3")
    repository.vacuum()

    assert checkpoint.mode == "passive"
    assert checkpoint.busy == 0
    assert backup_path.is_file()
    with closing(connect_database(backup_path)) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 20


def test_conditional_vacuum_uses_window_backup_space_and_threshold(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        retention_truncate_checkpoint_enabled=True,
        retention_maintenance_window_start_hour_utc=_NOW.hour,
        retention_vacuum_enabled=True,
        retention_vacuum_freelist_ratio_threshold=0.2,
        retention_vacuum_min_freelist_pages=10,
        retention_vacuum_free_space_multiplier=3,
        retention_vacuum_backup_dir=tmp_path / "backups",
    )
    initialize_database(settings.database_path)
    metrics = RetentionDatabaseMetrics(
        database_bytes=1_000,
        wal_bytes=100,
        shm_bytes=50,
        page_count=100,
        page_size=4_096,
        freelist_count=30,
    )

    class MaintenanceRepository:
        def __init__(self) -> None:
            self.database_path = settings.database_path
            self.operations: list[str] = []

        def metrics(self) -> RetentionDatabaseMetrics:
            return metrics

        def delete_batch(self, **_: object) -> RetentionCounts:
            return RetentionCounts()

        def activity(self, *, now: datetime) -> DatabaseActivity:
            del now
            self.operations.append("activity")
            return DatabaseActivity(0, 0, 0)

        def checkpoint(self, *, mode: str) -> WalCheckpointResult:
            self.operations.append(f"checkpoint:{mode}")
            return WalCheckpointResult(mode, 0, 4, 4)

        def online_backup(self, destination: Path) -> Path:
            self.operations.append("backup")
            destination.write_bytes(b"backup")
            return destination

        def vacuum(self) -> None:
            self.operations.append("vacuum")

    repository = MaintenanceRepository()
    worker = RetentionWorker(
        settings,
        repository=repository,  # type: ignore[arg-type]
        owner_id="backend-a",
        now_factory=lambda: _NOW,
        disk_free_factory=lambda _: 10_000,
    )

    result = worker.run_once(dry_run=False)

    assert result.checkpoint == WalCheckpointResult("truncate", 0, 4, 4)
    assert result.vacuum is not None
    assert result.vacuum.status is VacuumRunStatus.COMPLETED
    assert result.vacuum.freelist_ratio == 0.3
    assert result.vacuum.required_free_bytes == 3_000
    assert result.vacuum.backup_path is not None
    assert result.vacuum.backup_path.is_file()
    assert repository.operations.index("backup") < repository.operations.index("vacuum")
    assert repository.operations.count("vacuum") == 1
    assert repository.operations[-1] == "checkpoint:passive"


@pytest.mark.parametrize(
    "activity",
    [
        DatabaseActivity(1, 0, 0),
        DatabaseActivity(0, 1, 0),
        DatabaseActivity(0, 0, 1),
    ],
)
def test_conditional_vacuum_avoids_active_analysis_and_hiperwall_work(
    tmp_path: Path,
    activity: DatabaseActivity,
) -> None:
    settings = _settings(
        tmp_path,
        retention_truncate_checkpoint_enabled=True,
        retention_maintenance_window_start_hour_utc=_NOW.hour,
        retention_vacuum_enabled=True,
        retention_vacuum_freelist_ratio_threshold=0.2,
        retention_vacuum_min_freelist_pages=10,
    )
    initialize_database(settings.database_path)
    metrics = RetentionDatabaseMetrics(1_000, 0, 0, 100, 4_096, 30)

    class ActiveRepository:
        database_path = settings.database_path
        vacuum_called = False

        def metrics(self) -> RetentionDatabaseMetrics:
            return metrics

        def delete_batch(self, **_: object) -> RetentionCounts:
            return RetentionCounts()

        def activity(self, *, now: datetime) -> DatabaseActivity:
            del now
            return activity

        def checkpoint(self, *, mode: str) -> WalCheckpointResult:
            return WalCheckpointResult(mode, 0, 0, 0)

        def vacuum(self) -> None:
            self.vacuum_called = True

    repository = ActiveRepository()
    result = RetentionWorker(
        settings,
        repository=repository,  # type: ignore[arg-type]
        now_factory=lambda: _NOW,
    ).run_once(dry_run=False)

    assert result.checkpoint is not None
    assert result.checkpoint.mode == "passive"
    assert result.vacuum is not None
    assert result.vacuum.status is VacuumRunStatus.ACTIVE_WORK
    assert result.vacuum.active_work == activity
    assert repository.vacuum_called is False


def test_truncate_checkpoint_downgrades_to_passive_outside_window(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        retention_truncate_checkpoint_enabled=True,
        retention_maintenance_window_start_hour_utc=(_NOW.hour + 1) % 24,
        retention_maintenance_window_duration_minutes=30,
    )
    initialize_database(settings.database_path)

    result = RetentionWorker(settings, now_factory=lambda: _NOW).run_once(dry_run=False)

    assert result.checkpoint is not None
    assert result.checkpoint.mode == "passive"


def test_conditional_vacuum_skips_below_threshold_and_without_disk_space(
    tmp_path: Path,
) -> None:
    class PolicyRepository:
        def __init__(self, settings: Settings, metrics: RetentionDatabaseMetrics) -> None:
            self.database_path = settings.database_path
            self._metrics = metrics
            self.backup_called = False

        def metrics(self) -> RetentionDatabaseMetrics:
            return self._metrics

        def delete_batch(self, **_: object) -> RetentionCounts:
            return RetentionCounts()

        def activity(self, *, now: datetime) -> DatabaseActivity:
            del now
            return DatabaseActivity(0, 0, 0)

        def checkpoint(self, *, mode: str) -> WalCheckpointResult:
            return WalCheckpointResult(mode, 0, 0, 0)

        def online_backup(self, destination: Path) -> Path:
            self.backup_called = True
            return destination

    settings = _settings(
        tmp_path,
        retention_maintenance_window_start_hour_utc=_NOW.hour,
        retention_vacuum_enabled=True,
        retention_vacuum_freelist_ratio_threshold=0.25,
        retention_vacuum_min_freelist_pages=10,
        retention_vacuum_backup_dir=tmp_path / "backups",
    )
    initialize_database(settings.database_path)

    below = PolicyRepository(
        settings,
        RetentionDatabaseMetrics(1_000, 0, 0, 100, 4_096, 20),
    )
    below_result = RetentionWorker(
        settings,
        repository=below,  # type: ignore[arg-type]
        now_factory=lambda: _NOW,
    ).run_once(dry_run=False)
    assert below_result.vacuum is not None
    assert below_result.vacuum.status is VacuumRunStatus.BELOW_THRESHOLD

    no_space = PolicyRepository(
        settings,
        RetentionDatabaseMetrics(1_000, 0, 0, 100, 4_096, 30),
    )
    no_space_result = RetentionWorker(
        settings,
        repository=no_space,  # type: ignore[arg-type]
        now_factory=lambda: _NOW,
        disk_free_factory=lambda _: 2_999,
    ).run_once(dry_run=False)
    assert no_space_result.vacuum is not None
    assert no_space_result.vacuum.status is VacuumRunStatus.INSUFFICIENT_DISK_SPACE
    assert no_space.backup_called is False


def test_retention_manual_parser_requires_explicit_apply_for_deletion() -> None:
    parser = build_argument_parser()

    assert parser.parse_args([]).dry_run is None
    assert parser.parse_args(["--dry-run"]).dry_run is True
    assert parser.parse_args(["--apply"]).dry_run is False


def test_application_starts_and_stops_injected_retention_worker(tmp_path: Path) -> None:
    class FakeRetentionWorker:
        def __init__(self) -> None:
            self.started = 0
            self.stopped = 0

        def start(self) -> None:
            self.started += 1

        def stop(self) -> None:
            self.stopped += 1

    settings = _settings(tmp_path)
    worker = FakeRetentionWorker()

    with TestClient(
        create_app(settings, retention_worker=worker)  # type: ignore[arg-type]
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.retention_worker is worker

    assert worker.started == 1
    assert worker.stopped == 1
