import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import (
    AnalysisLeaseRepository,
    AnalysisRunStatus,
    AnalysisWorkerFailureRepository,
    CameraRepository,
    DetectionRepository,
    RuleRepository,
    connect_database,
    initialize_database,
)
from cctv.main import create_app
from cctv.workers import (
    AnalysisSupervisor,
    AnalysisSupervisorSnapshot,
    AnalysisWorkerSnapshot,
    AnalysisWorkerStatus,
)


class FakeProcess:
    _next_pid = 100

    def __init__(self) -> None:
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.stdout: tuple[str, ...] = ()
        self.return_code: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = -15

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.return_code is None:
            self.return_code = 0
        return self.return_code


class HangingFakeProcess(FakeProcess):
    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if self.return_code is None:
            raise subprocess.TimeoutExpired("analysis-worker", timeout)
        return self.return_code


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "test",
        "app_mode": "live",
        "hiperwall_executor_enabled": False,
        "database_path": tmp_path / "cctv.db",
        "model_path": model_path,
        "log_path": tmp_path / "logs" / "cctv.jsonl",
        "rtsp_worker_url": "rtsp://mediamtx:8554/camera",
        "analysis_supervisor_enabled": True,
        "analysis_supervisor_max_workers": 2,
        "analysis_supervisor_restart_base_seconds": 2,
        "analysis_supervisor_restart_max_seconds": 8,
    }
    values.update(overrides)
    return Settings(**values)


def _create_color_rule(repository: RuleRepository, source_name: str, name: str):
    return repository.create_rule(
        name=name,
        source_name=source_name,
        rule_type="visual_color",
        class_name="person",
        geometry={},
        parameters={
            "target_color": "green",
            "window_size": 5,
            "minimum_matches": 3,
            "minimum_color_confidence": 0.35,
            "cooldown_seconds": 30,
        },
    )


@dataclass
class _WatchdogScenario:
    clock: FakeClock
    base_time: datetime
    settings: Settings
    supervisor: AnalysisSupervisor
    process: HangingFakeProcess
    camera_id: int
    run_id: str

    def emit_heartbeat(
        self,
        *,
        at: float,
        stage: str,
        stage_started_at: datetime,
    ) -> None:
        self.clock.value = at
        self.supervisor._consume_worker_record(
            self.camera_id,
            self.process,
            {
                "event": "rtsp_worker_heartbeat",
                "stage": stage,
                "stage_started_at": stage_started_at.isoformat(),
                "last_frame_at": self.base_time.isoformat(),
            },
        )


def _create_watchdog_scenario(tmp_path: Path) -> _WatchdogScenario:
    clock = FakeClock()
    base_time = datetime(2026, 8, 21, 8, tzinfo=UTC)
    settings = _settings(
        tmp_path,
        analysis_worker_heartbeat_interval_seconds=1,
        analysis_supervisor_stale_timeout_seconds=10,
        analysis_supervisor_initial_grace_seconds=30,
        analysis_supervisor_reconnect_stale_timeout_seconds=30,
        analysis_supervisor_shutdown_timeout_seconds=1,
    )
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    camera = cameras.create_camera(name="Lobby", stream_path="lobby")
    _create_color_rule(rules, camera.stream_path, "Rule")
    process = HangingFakeProcess()

    def launch(command: list[str], environment: dict[str, str]) -> HangingFakeProcess:
        del command, environment
        return process

    supervisor = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=launch,
        monotonic_clock=clock,
        now_factory=lambda: base_time + timedelta(seconds=clock.value),
        owner_id="backend-a",
    )
    supervisor.reconcile_once()
    run_id = supervisor.snapshot().items[0].analysis_run_id
    assert run_id is not None
    DetectionRepository(settings.database_path).create_analysis_run(
        source_type="rtsp",
        source_name="lobby",
        camera_id=camera.id,
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
    with connect_database(settings.database_path) as connection:
        connection.execute(
            """
            INSERT INTO tracks (
                analysis_run_id, track_id, class_id, class_name,
                first_sample_index, last_sample_index,
                first_seen_timestamp_seconds, last_seen_timestamp_seconds,
                observation_count, max_confidence, is_active
            ) VALUES (?, 1, 0, ?, 0, 0, 0, 0, 1, 0.9, 1)
            """,
            (run_id, "person"),
        )
        connection.commit()
    return _WatchdogScenario(
        clock=clock,
        base_time=base_time,
        settings=settings,
        supervisor=supervisor,
        process=process,
        camera_id=camera.id,
        run_id=run_id,
    )


def test_supervisor_starts_reloads_and_stops_one_worker_per_camera(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, frame_persistence_mode="events")
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    camera = cameras.create_camera(name="Lobby", stream_path="lobby/camera")
    rule = _create_color_rule(rules, camera.stream_path, "Green shirt")
    launches: list[tuple[list[str], dict[str, str], FakeProcess]] = []

    def launch(command: list[str], environment: dict[str, str]) -> FakeProcess:
        process = FakeProcess()
        launches.append((command, environment, process))
        return process

    monkeypatch.setenv("HIPERWALL_TOKEN", "must-not-reach-analysis-worker")
    supervisor = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=launch,
    )

    supervisor.reconcile_once()

    first = launches[0]
    snapshot = supervisor.snapshot()
    assert snapshot.active_workers == 1
    assert snapshot.items[0].status is AnalysisWorkerStatus.STARTING
    assert snapshot.items[0].lease_owned is True
    assert snapshot.items[0].rule_count == 1
    assert "--analyze" in first[0]
    assert not any("rtsp://" in argument for argument in first[0])
    assert first[1]["CCTV_RTSP_WORKER_URL"] == "rtsp://mediamtx:8554/lobby/camera"
    assert first[1]["CCTV_APP_MODE"] == "live"
    assert first[1]["CCTV_AI_DEVICE"] == "cpu"
    assert first[1]["CCTV_AI_ALLOW_CPU_FALLBACK"] == "false"
    assert first[1]["CCTV_AI_CUDA_DEVICE_ID"] == "0"
    assert "CCTV_AI_CUDA_GPU_MEM_LIMIT_MB" not in first[1]
    assert first[1]["CCTV_FRAME_PERSISTENCE_MODE"] == "events"
    assert first[1]["CCTV_HIPERWALL_EXECUTOR_ENABLED"] == "false"
    assert "HIPERWALL_TOKEN" not in first[1]

    rules.update_rule(
        rule.id,
        name=rule.name,
        source_name=rule.source_name,
        rule_type=rule.rule_type,
        class_name=rule.class_name,
        geometry=rule.geometry,
        parameters={**rule.parameters, "minimum_matches": 4},
        enabled=True,
    )
    supervisor.reconcile_once()

    assert len(launches) == 2
    assert first[2].terminated is True
    assert (
        supervisor.snapshot().items[0].configuration_revision
        != snapshot.items[0].configuration_revision
    )

    cameras.update_camera(camera.id, {"enabled": False})
    supervisor.reconcile_once()

    final = supervisor.snapshot().items[0]
    assert launches[1][2].terminated is True
    assert final.status is AnalysisWorkerStatus.STOPPED
    assert final.desired is False
    assert final.rule_count == 0


def test_supervisor_enforces_capacity_and_restarts_after_backoff(tmp_path: Path) -> None:
    settings = _settings(tmp_path, analysis_supervisor_max_workers=1)
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    first_camera = cameras.create_camera(name="First", stream_path="first")
    second_camera = cameras.create_camera(name="Second", stream_path="second")
    _create_color_rule(rules, first_camera.stream_path, "First rule")
    _create_color_rule(rules, second_camera.stream_path, "Second rule")
    clock = FakeClock()
    launches: list[FakeProcess] = []

    def launch(command: list[str], environment: dict[str, str]) -> FakeProcess:
        del command, environment
        process = FakeProcess()
        launches.append(process)
        return process

    supervisor = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=launch,
        monotonic_clock=clock,
    )
    supervisor.reconcile_once()

    assert len(launches) == 1
    assert supervisor.snapshot().active_workers == 1
    assert supervisor.snapshot().items[1].last_message == "waiting for worker capacity"

    launches[0].return_code = 3
    supervisor.reconcile_once()

    failed = supervisor.snapshot().items[0]
    assert failed.status is AnalysisWorkerStatus.FAILED
    assert failed.restart_count == 1
    assert failed.next_restart_at is not None
    assert len(launches) == 2

    cameras.update_camera(second_camera.id, {"enabled": False})
    supervisor.reconcile_once()
    clock.value = 2
    supervisor.reconcile_once()

    assert len(launches) == 3
    assert supervisor.snapshot().items[0].status is AnalysisWorkerStatus.STARTING


def test_analysis_worker_status_api_reports_disabled_and_running_states(tmp_path: Path) -> None:
    disabled_settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=tmp_path / "disabled.db",
        model_path=tmp_path / "missing.onnx",
        log_path=tmp_path / "disabled.jsonl",
    )
    with TestClient(create_app(disabled_settings)) as client:
        disabled = client.get("/analysis-workers")
        missing = client.get("/analysis-workers/1")

    assert disabled.status_code == 200
    assert disabled.json() == {
        "enabled": False,
        "running": False,
        "max_workers": 2,
        "active_workers": 0,
        "last_reconciled_at": None,
        "items": [],
    }
    assert missing.status_code == 404

    now = datetime(2026, 8, 21, tzinfo=UTC)
    worker = AnalysisWorkerSnapshot(
        camera_id=1,
        camera_name="Lobby",
        stream_path="camera",
        status=AnalysisWorkerStatus.RUNNING,
        desired=True,
        lease_owned=True,
        rule_count=2,
        configuration_revision="abc123",
        analysis_run_id="run-1",
        processed_samples=12,
        connection_count=1,
        reconnect_count=0,
        restart_count=0,
        started_at=now,
        last_frame_at=now,
        last_event_at=None,
        completed_at=None,
        next_restart_at=None,
        last_error_type=None,
        last_message="RTSP stream connected",
    )

    class FakeSupervisor:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def snapshot(self) -> AnalysisSupervisorSnapshot:
            return AnalysisSupervisorSnapshot(
                enabled=True,
                running=True,
                max_workers=2,
                active_workers=1,
                last_reconciled_at=now,
                items=(worker,),
            )

    enabled_settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=tmp_path / "enabled.db",
        model_path=tmp_path / "missing.onnx",
        log_path=tmp_path / "enabled.jsonl",
    )
    with TestClient(
        create_app(enabled_settings, analysis_supervisor=FakeSupervisor())  # type: ignore[arg-type]
    ) as client:
        listed = client.get("/analysis-workers")
        detail = client.get("/analysis-workers/1")

    assert listed.status_code == 200
    assert listed.json()["active_workers"] == 1
    assert listed.json()["items"][0]["status"] == "running"
    assert listed.json()["items"][0]["cpu_fallback"] is False
    assert detail.status_code == 200
    assert detail.json()["analysis_run_id"] == "run-1"


def test_worker_output_updates_progress_and_reconnect_state(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    camera = cameras.create_camera(name="Lobby", stream_path="camera")
    _create_color_rule(rules, camera.stream_path, "Rule")
    reported_frame_at = datetime(2026, 8, 21, 1, 2, 3, 456000, tzinfo=UTC)
    records = (
        json.dumps(
            {
                "event": "detector_runtime_resolved",
                "requested_device": "cuda",
                "effective_device": "cpu",
                "execution_provider": "OpenCVDNNCPU",
                "cpu_fallback": True,
                "fallback_reason": "CUDAExecutionProvider is unavailable",
            }
        ),
        json.dumps({"event": "analysis_run_created", "analysis_run_id": "run-1"}),
        json.dumps({"event": "rtsp_source_connected", "connection_count": 1}),
        json.dumps(
            {
                "event": "rtsp_worker_progress",
                "status": "running",
                "processed_samples": 5,
                "last_frame_at": "2026-08-21T01:02:03.456Z",
            }
        ),
        json.dumps({"event": "rule_event_emitted"}),
        json.dumps({"event": "rtsp_source_reconnect_scheduled", "reconnect_count": 1}),
    )

    def launch(command: list[str], environment: dict[str, str]) -> FakeProcess:
        del command, environment
        process = FakeProcess()
        process.stdout = records
        return process

    supervisor = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=launch,
    )
    supervisor.reconcile_once()

    for _ in range(100):
        snapshot = supervisor.snapshot().items[0]
        if snapshot.reconnect_count == 1:
            break
        sleep(0.01)

    assert snapshot.analysis_run_id == "run-1"
    assert snapshot.requested_device == "cuda"
    assert snapshot.effective_device == "cpu"
    assert snapshot.execution_provider == "OpenCVDNNCPU"
    assert snapshot.cpu_fallback is True
    assert snapshot.fallback_reason == "CUDAExecutionProvider is unavailable"
    assert snapshot.processed_samples == 5
    assert snapshot.status is AnalysisWorkerStatus.RECONNECTING
    assert snapshot.last_frame_at == reported_frame_at
    assert snapshot.last_event_at is not None


def test_analysis_lease_is_exclusive_and_can_be_taken_after_expiry(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    camera = CameraRepository(settings.database_path).create_camera(
        name="Lobby",
        stream_path="camera",
    )
    leases = AnalysisLeaseRepository(settings.database_path)
    now = datetime(2026, 8, 21, tzinfo=UTC)

    assert leases.acquire(camera.id, "backend-a", lease_seconds=20, now=now) is True
    assert leases.acquire(camera.id, "backend-b", lease_seconds=20, now=now) is False
    assert (
        leases.acquire(
            camera.id,
            "backend-b",
            lease_seconds=20,
            now=now + timedelta(seconds=21),
        )
        is True
    )
    assert leases.release(camera.id, "backend-a") is False
    assert leases.release(camera.id, "backend-b") is True


def test_two_supervisors_do_not_start_duplicate_camera_workers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    camera = cameras.create_camera(name="Lobby", stream_path="camera")
    _create_color_rule(rules, camera.stream_path, "Rule")
    first_launches: list[FakeProcess] = []
    second_launches: list[FakeProcess] = []

    def first_launch(command: list[str], environment: dict[str, str]) -> FakeProcess:
        del command, environment
        process = FakeProcess()
        first_launches.append(process)
        return process

    def second_launch(command: list[str], environment: dict[str, str]) -> FakeProcess:
        del command, environment
        process = FakeProcess()
        second_launches.append(process)
        return process

    first = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=first_launch,
        owner_id="backend-a",
    )
    second = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=second_launch,
        owner_id="backend-b",
    )

    first.reconcile_once()
    second.reconcile_once()

    assert len(first_launches) == 1
    assert second_launches == []
    assert second.snapshot().items[0].lease_owned is False
    assert second.snapshot().items[0].last_message == "another Backend instance owns this camera"

    first.stop()
    second.reconcile_once()

    assert len(second_launches) == 1
    assert second.snapshot().items[0].lease_owned is True
    second.stop()


@pytest.mark.parametrize("stage", ("inference", "database_save"))
def test_watchdog_restarts_stalled_stage_despite_fresh_heartbeat(
    tmp_path: Path,
    stage: str,
) -> None:
    scenario = _create_watchdog_scenario(tmp_path)
    stage_started_at = scenario.base_time + timedelta(seconds=1)
    scenario.emit_heartbeat(
        at=1,
        stage=stage,
        stage_started_at=stage_started_at,
    )
    scenario.emit_heartbeat(
        at=10.9,
        stage=stage,
        stage_started_at=stage_started_at,
    )
    scenario.supervisor.reconcile_once()
    assert scenario.process.terminated is False
    assert scenario.process.killed is False

    scenario.clock.value = 11.1
    scenario.supervisor.reconcile_once()

    expected_failure = f"{stage}_stalled"
    snapshot = scenario.supervisor.snapshot().items[0]
    assert scenario.process.terminated is True
    assert scenario.process.killed is True
    assert snapshot.status is AnalysisWorkerStatus.FAILED
    assert snapshot.current_error_type == expected_failure
    assert snapshot.last_failure_type == expected_failure
    assert snapshot.last_failure_run_id == scenario.run_id
    assert snapshot.next_restart_at is not None
    with connect_database(scenario.settings.database_path) as connection:
        run = connection.execute(
            "SELECT status, completion_reason FROM analysis_runs WHERE id = ?",
            (scenario.run_id,),
        ).fetchone()
        active_tracks = connection.execute(
            "SELECT COUNT(*) FROM tracks WHERE analysis_run_id = ? AND is_active = 1",
            (scenario.run_id,),
        ).fetchone()[0]
    assert run["status"] == AnalysisRunStatus.INTERRUPTED
    assert run["completion_reason"] == f"watchdog_{expected_failure}"
    assert active_tracks == 0


def test_failure_summary_survives_supervisor_recreation_and_limits_restart_storm(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    base_time = datetime(2026, 8, 21, 9, tzinfo=UTC)
    settings = _settings(
        tmp_path,
        analysis_supervisor_max_restarts_in_window=2,
        analysis_supervisor_restart_window_seconds=60,
    )
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    camera = cameras.create_camera(name="Lobby", stream_path="lobby")
    _create_color_rule(rules, camera.stream_path, "Rule")
    processes: list[FakeProcess] = []

    def launch(command: list[str], environment: dict[str, str]) -> FakeProcess:
        del command, environment
        process = FakeProcess()
        processes.append(process)
        return process

    supervisor = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=launch,
        monotonic_clock=clock,
        now_factory=lambda: base_time + timedelta(seconds=clock.value),
        owner_id="backend-a",
    )
    supervisor.reconcile_once()
    processes[0].return_code = 3
    supervisor.reconcile_once()
    clock.value = 2
    supervisor.reconcile_once()
    processes[1].return_code = 4
    supervisor.reconcile_once()

    rate_limited = supervisor.snapshot().items[0]
    assert rate_limited.restart_count == 2
    assert rate_limited.next_restart_at == base_time + timedelta(seconds=60)
    clock.value = 59
    supervisor.reconcile_once()
    assert len(processes) == 2

    supervisor.lease_repository.release_all("backend-a")
    recreated = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        failure_repository=AnalysisWorkerFailureRepository(settings.database_path),
        process_factory=launch,
        monotonic_clock=clock,
        now_factory=lambda: base_time + timedelta(seconds=clock.value),
        owner_id="backend-b",
    )
    recreated.reconcile_once()
    summary = recreated.snapshot().items[0]
    assert summary.restart_count == 2
    assert summary.current_error_type is None
    assert summary.last_failure_type == "ProcessExit4"
    assert summary.last_failure_message == "analysis worker exited unexpectedly"


def test_backend_shutdown_marks_the_current_run_interrupted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    initialize_database(settings.database_path)
    cameras = CameraRepository(settings.database_path)
    rules = RuleRepository(settings.database_path)
    camera = cameras.create_camera(name="Lobby", stream_path="lobby")
    _create_color_rule(rules, camera.stream_path, "Rule")
    process = FakeProcess()

    supervisor = AnalysisSupervisor(
        settings,
        camera_repository=cameras,
        rule_repository=rules,
        process_factory=lambda command, environment: process,
        owner_id="backend-a",
    )
    supervisor.reconcile_once()
    run_id = supervisor.snapshot().items[0].analysis_run_id
    assert run_id is not None
    DetectionRepository(settings.database_path).create_analysis_run(
        source_type="rtsp",
        source_name="lobby",
        camera_id=camera.id,
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

    supervisor.stop()

    with connect_database(settings.database_path) as connection:
        run = connection.execute(
            "SELECT status, completion_reason FROM analysis_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    assert process.terminated is True
    assert run["status"] == AnalysisRunStatus.INTERRUPTED
    assert run["completion_reason"] == "backend_shutdown"
