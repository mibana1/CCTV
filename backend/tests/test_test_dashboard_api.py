from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from cctv.cameras import CameraCredentialCipher, CameraManagementService, MediaMtxPathStatus
from cctv.core.settings import Settings
from cctv.dashboard import (
    DashboardEvent,
    SnapshotInfo,
)
from cctv.dashboard import TestSessionSnapshot as DashboardSessionSnapshot
from cctv.dashboard import TestSessionStatus as DashboardSessionStatus
from cctv.db import (
    CameraRepository,
    DetectionRepository,
    FaceMatchEventInput,
    FaceMatchRepository,
    initialize_database,
)
from cctv.main import create_app


class FakeSessionManager:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_file = snapshot_path
        self.session = DashboardSessionSnapshot(
            id="session-1",
            status=DashboardSessionStatus.STARTING,
            source_name="camera-1",
            sample_fps=2,
            max_samples=20,
            save_snapshots=True,
            match_faces=True,
            analysis_run_id=None,
            processed_samples=0,
            connection_count=0,
            reconnect_count=0,
            elapsed_seconds=0,
            cpu_percent=None,
            memory_mib=None,
            peak_cpu_percent=None,
            peak_memory_mib=None,
            snapshot_count=1,
            started_at=datetime.now(UTC),
            completed_at=None,
            last_message=None,
            error_type=None,
            result_summary=None,
        )

    def start(self, **kwargs) -> DashboardSessionSnapshot:
        self.session = replace(
            self.session,
            sample_fps=kwargs["sample_fps"],
            max_samples=kwargs["max_samples"],
            save_snapshots=kwargs["save_snapshots"],
            match_faces=kwargs["match_faces"],
        )
        return self.session

    def stop(self, session_id: str) -> DashboardSessionSnapshot:
        assert session_id == self.session.id
        self.session = replace(
            self.session,
            status=DashboardSessionStatus.STOPPED,
            completed_at=datetime.now(UTC),
        )
        return self.session

    def get(self, session_id: str) -> DashboardSessionSnapshot:
        if session_id != self.session.id:
            raise LookupError(session_id)
        return self.session

    def list(self) -> tuple[DashboardSessionSnapshot, ...]:
        return (self.session,)

    def events(self, session_id: str, *, after: int = 0) -> tuple[DashboardEvent, ...]:
        if after > 0:
            return ()
        return (
            DashboardEvent(
                sequence=1,
                event="session_finished",
                timestamp=datetime.now(UTC),
                data={"status": "completed"},
            ),
        )

    def snapshots(self, session_id: str) -> tuple[SnapshotInfo, ...]:
        assert session_id == self.session.id
        return (SnapshotInfo(name=self.snapshot_file.name, sample_index=1, size_bytes=3),)

    def snapshot_path(self, session_id: str, name: str) -> Path:
        assert session_id == self.session.id
        if name != self.snapshot_file.name:
            raise FileNotFoundError(name)
        return self.snapshot_file

    def shutdown(self) -> None:
        pass


class FakeMediaMtx:
    def __init__(self) -> None:
        self.paths: dict[str, str] = {}

    def upsert_rtsp_path(
        self,
        *,
        stream_path: str,
        source_url: str,
        source_on_demand: bool,
    ) -> None:
        self.paths[stream_path] = source_url

    def delete_path(self, stream_path: str) -> None:
        self.paths.pop(stream_path, None)

    def get_path_status(self, stream_path: str) -> MediaMtxPathStatus | None:
        return (
            MediaMtxPathStatus(online=False, available=False)
            if stream_path in self.paths
            else None
        )

    def close(self) -> None:
        pass


def _settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        app_mode="dry_run",
        test_dashboard_enabled=enabled,
        database_path=tmp_path / "cctv.db",
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "cctv.jsonl",
    )


def _camera_manager(settings: Settings) -> CameraManagementService:
    initialize_database(settings.database_path)
    return CameraManagementService(
        CameraRepository(settings.database_path),
        CameraCredentialCipher(Fernet.generate_key()),
        FakeMediaMtx(),  # type: ignore[arg-type]
    )


def test_dashboard_controls_sessions_and_serves_snapshots(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample_000001_frame_000000001_t000000500ms.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)

    with TestClient(create_app(_settings(tmp_path), test_session_manager=manager)) as client:
        page = client.get("/test-dashboard")
        started = client.post(
            "/test-sessions",
            json={
                "sample_fps": 3,
                "max_samples": 30,
                "save_snapshots": True,
                "match_faces": True,
            },
        )
        sessions = client.get("/test-sessions")
        snapshots = client.get("/test-sessions/session-1/snapshots")
        image = client.get(
            "/test-sessions/session-1/snapshots/"
            "sample_000001_frame_000000001_t000000500ms.jpg"
        )
        stopped = client.post("/test-sessions/session-1/stop")
        event_stream = client.get("/test-sessions/session-1/events")

    assert page.status_code == 200
    assert "RTSP 얼굴 인식 테스트" in page.text
    assert "카메라 등록" in page.text
    assert started.status_code == 201
    assert started.json()["sample_fps"] == 3
    assert started.json()["max_samples"] == 30
    assert sessions.json()["items"][0]["id"] == "session-1"
    assert snapshots.json()["items"][0]["sample_index"] == 1
    assert image.content == b"jpg"
    assert stopped.status_code == 202
    assert stopped.json()["status"] == "stopped"
    assert event_stream.headers["content-type"].startswith("text/event-stream")
    assert "event: session_finished" in event_stream.text


def test_dashboard_page_stabilizes_result_refreshes(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)

    with TestClient(create_app(_settings(tmp_path), test_session_manager=manager)) as client:
        page = client.get("/test-dashboard")

    assert page.status_code == 200
    assert "refreshInFlight: null" in page.text
    assert "if (state.refreshInFlight) return state.refreshInFlight;" in page.text
    assert "stopRefreshTimer();" in page.text
    assert 'document.getElementById("gallery").replaceChildren(fragment);' in page.text
    assert "person-instance-summary" in page.text
    assert "item.person_instance_id" in page.text
    assert (
        'document.getElementById("face-results").replaceChildren(...replacement.children);'
        in page.text
    )


def test_dashboard_camera_api_manages_preview_registrations(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)
    settings = _settings(tmp_path)
    camera_manager = _camera_manager(settings)

    with TestClient(
        create_app(
            settings,
            test_session_manager=manager,
            camera_manager=camera_manager,
        )
    ) as client:
        created = client.post(
            "/test-cameras",
            json={
                "name": "Main entrance",
                "location": "Lobby",
                "rtsp_url": "rtsp://camera.local:554/building/entrance?profile=main",
                "username": "operator",
                "password": "top-secret",
            },
        )
        camera_id = created.json()["id"]
        cameras = client.get("/test-cameras")
        camera = client.get(f"/test-cameras/{camera_id}")
        updated = client.patch(
            f"/test-cameras/{camera_id}",
            json={"name": "Front door", "enabled": False},
        )
        embedded_credentials = client.post(
            "/test-cameras",
            json={
                "name": "Unsafe",
                "rtsp_url": "rtsp://user:pass@camera.local/stream",
            },
        )
        blank_name = client.post(
            "/test-cameras",
            json={"name": "   ", "rtsp_url": "rtsp://camera.local/other"},
        )
        status_response = client.get(f"/test-cameras/{camera_id}/status")
        deleted = client.delete(f"/test-cameras/{camera_id}")
        missing = client.get(f"/test-cameras/{camera_id}")

    assert created.status_code == 201
    assert created.json()["stream_path"].startswith("cam-")
    assert created.json()["preview_url"].startswith("http://127.0.0.1:18888/cam-")
    assert created.json()["rtsp_endpoint"] == "rtsp://camera.local:554/building/entrance"
    assert created.json()["credentials_configured"] is True
    assert "top-secret" not in created.text
    assert cameras.json()["total"] == 1
    assert camera.json()["location"] == "Lobby"
    assert updated.json()["name"] == "Front door"
    assert updated.json()["enabled"] is False
    assert embedded_credentials.status_code == 422
    assert blank_name.status_code == 422
    assert status_response.status_code == 200
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_dashboard_face_event_api_returns_persisted_results(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)
    settings = _settings(tmp_path)

    with TestClient(create_app(settings, test_session_manager=manager)) as client:
        detection_repository = DetectionRepository(settings.database_path)
        run = detection_repository.create_analysis_run(
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
        FaceMatchRepository(settings.database_path).save_event(
            run.id,
            FaceMatchEventInput(
                source_index=10,
                sample_index=1,
                source_timestamp_seconds=0.5,
                face_index=0,
                track_id=7,
                face_x=10,
                face_y=20,
                face_width=30,
                face_height=40,
                detection_confidence=0.9,
                match_status="matched",
                rejection_reason=None,
                identity_id="emp-001",
                external_id="EMP-001",
                display_name="Test Person",
                best_candidate_identity_id="emp-001",
                best_candidate_external_id="EMP-001",
                best_similarity=0.61,
                second_best_similarity=None,
                similarity_threshold=0.45,
                minimum_margin=0.05,
            ),
        )

        events = client.get("/face-match-events", params={"analysis_run_id": run.id})
        summary = client.get(f"/analysis-runs/{run.id}/face-match-summary")

    assert events.status_code == 200
    assert events.json()["items"][0]["face_bounds"] == {
        "x": 10,
        "y": 20,
        "width": 30,
        "height": 40,
    }
    assert events.json()["items"][0]["external_id"] == "EMP-001"
    assert summary.json()["matched_faces"] == 1
    assert summary.json()["unknown_faces"] == 0


def test_dashboard_is_denied_when_feature_is_disabled(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)

    with TestClient(
        create_app(_settings(tmp_path, enabled=False), test_session_manager=manager)
    ) as client:
        response = client.get("/test-dashboard")

    assert response.status_code == 403


def test_dashboard_rejects_non_local_host_and_live_mode(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)
    dry_run_settings = _settings(tmp_path)
    live_settings = Settings(
        _env_file=None,
        app_env="development",
        app_mode="live",
        test_dashboard_enabled=True,
        hiperwall_base_url="http://hiperwall.local",
        database_path=tmp_path / "live.db",
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "live.jsonl",
    )

    with TestClient(
        create_app(dry_run_settings, test_session_manager=manager)
    ) as dry_run_client:
        remote_response = dry_run_client.get(
            "/test-dashboard",
            headers={"host": "192.168.1.20:8000"},
        )
    with TestClient(create_app(live_settings, test_session_manager=manager)) as live_client:
        live_response = live_client.get("/test-dashboard")

    assert remote_response.status_code == 403
    assert live_response.status_code == 403
