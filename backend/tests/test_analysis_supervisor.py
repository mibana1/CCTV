import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import (
    AnalysisLeaseRepository,
    CameraRepository,
    RuleRepository,
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


def test_supervisor_starts_reloads_and_stops_one_worker_per_camera(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
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
        json.dumps({"event": "rtsp_worker_progress", "processed_samples": 5}),
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
    assert snapshot.last_frame_at is not None
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
