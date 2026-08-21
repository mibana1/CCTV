from __future__ import annotations

import base64
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from cctv.cameras import CameraCredentialCipher, CameraManagementService, MediaMtxPathStatus
from cctv.core.settings import Settings
from cctv.dashboard import (
    DashboardEvent,
    SessionConflictError,
    SnapshotInfo,
)
from cctv.dashboard import TestSessionSnapshot as DashboardSessionSnapshot
from cctv.dashboard import TestSessionStatus as DashboardSessionStatus
from cctv.db import (
    CameraRepository,
    DetectionRepository,
    FaceMatchEventInput,
    FaceMatchRepository,
    IdentityRepository,
    PersonInstanceRepository,
    initialize_database,
)
from cctv.identity import (
    ExtractedFaceEmbedding,
    FaceEmbeddingModelMetadata,
    FacePhotoError,
    FaceRegistrationService,
    TrackIdentityResolver,
)
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.main import create_app


class FakeSessionManager:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_file = snapshot_path
        self.last_start_kwargs = {}
        self.deleted = False
        self.session = DashboardSessionSnapshot(
            id="session-1",
            status=DashboardSessionStatus.STARTING,
            camera_id=1,
            stream_path="camera-1",
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
        self.last_start_kwargs = kwargs
        self.session = replace(
            self.session,
            camera_id=kwargs["camera_id"],
            stream_path=kwargs["stream_path"],
            source_name=kwargs["stream_path"],
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
        return () if self.deleted else (self.session,)

    def delete(self, session_id: str) -> None:
        assert session_id == self.session.id
        if self.session.status not in {
            DashboardSessionStatus.COMPLETED,
            DashboardSessionStatus.STOPPED,
            DashboardSessionStatus.FAILED,
        }:
            raise SessionConflictError("an active test session cannot be deleted")
        self.deleted = True

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


class FakeRegistrationExtractor:
    metadata = FaceEmbeddingModelMetadata()

    def extract_file(self, photo_path: str | Path) -> ExtractedFaceEmbedding:
        path = Path(photo_path)
        index = int(path.stem.rsplit("-", 1)[1])
        vector = [0.0] * self.metadata.dimension
        vector[0] = 1.0
        vector[index] = 0.1
        return ExtractedFaceEmbedding(
            vector=tuple(vector),
            detection_confidence=0.9 + index / 100,
        )


class FailingRegistrationExtractor(FakeRegistrationExtractor):
    def extract_file(self, photo_path: str | Path) -> ExtractedFaceEmbedding:
        if Path(photo_path).stem.endswith("02"):
            raise FacePhotoError("registration photo must contain exactly one face; found 0")
        return super().extract_file(photo_path)


def _settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        app_mode="dry_run",
        test_dashboard_enabled=enabled,
        database_path=tmp_path / "cctv.db",
        model_path=tmp_path / "model.onnx",
        identity_photo_dir=tmp_path / "identity-images",
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
    settings = _settings(tmp_path)

    with TestClient(create_app(settings, test_session_manager=manager)) as client:
        camera = CameraRepository(settings.database_path).create_camera(
            name="Test camera",
            stream_path="camera-test",
        )
        page = client.get("/test-dashboard")
        identity_page = client.get("/test-identities")
        color_event_page = client.get("/test-color-events")
        started = client.post(
            "/test-sessions",
            json={
                "camera_id": camera.id,
                "stream_path": camera.stream_path,
                "sample_fps": 3,
                "max_samples": 30,
                "save_snapshots": True,
                "match_faces": True,
            },
        )
        sessions = client.get("/test-sessions")
        snapshots = client.get("/test-sessions/session-1/snapshots")
        active_delete = client.delete("/test-sessions/session-1")
        image = client.get(
            "/test-sessions/session-1/snapshots/"
            "sample_000001_frame_000000001_t000000500ms.jpg"
        )
        stopped = client.post("/test-sessions/session-1/stop")
        event_stream = client.get("/test-sessions/session-1/events")
        deleted = client.delete("/test-sessions/session-1")
        sessions_after_delete = client.get("/test-sessions")

    assert page.status_code == 200
    assert "RTSP 얼굴 인식 테스트" in page.text
    assert 'href="/test-identities"' in page.text
    assert 'href="/test-color-events"' in page.text
    assert 'id="person-form"' not in page.text
    assert identity_page.status_code == 200
    assert "사람 등록 · CCTV 테스트" in identity_page.text
    assert 'id="person-form"' in identity_page.text
    assert 'href="/test-dashboard"' in identity_page.text
    assert 'api("/test-identities"' in identity_page.text
    assert 'method: "DELETE"' in identity_page.text
    assert "등록 사진과 얼굴 특징도 함께 삭제됩니다" in identity_page.text
    assert color_event_page.status_code == 200
    assert "색상 이벤트 · CCTV 테스트" in color_event_page.text
    assert 'id="color-rule-form"' in color_event_page.text
    assert 'rule_type: "visual_color"' in color_event_page.text
    assert 'target_color: document.getElementById("target-color").value' in color_event_page.text
    assert 'editingRuleId ? `/rules/${encodeURIComponent(editingRuleId)}` : "/rules"' in color_event_page.text
    assert "Hiperwall 연결 · 선택" in color_event_page.text
    assert 'id="hiperwall-reload"' in color_event_page.text
    assert 'api("/hiperwall/inventory")' in color_event_page.text
    assert 'id="hiperwall-content"' in color_event_page.text
    assert 'id="hiperwall-zone"' in color_event_page.text
    assert 'content_uuid: content?.uuid || contentValue' in color_event_page.text
    assert 'id="hiperwall-layout-mode"' in color_event_page.text
    assert 'id="hiperwall-x"' in color_event_page.text
    assert 'id="hiperwall-y"' in color_event_page.text
    assert 'id="hiperwall-width"' in color_event_page.text
    assert 'id="hiperwall-height"' in color_event_page.text
    assert "function applyZoneLayoutDefaults()" in color_event_page.text
    assert "return { mode, x, y, width, height };" in color_event_page.text
    assert 'test.textContent = "Hiperwall 테스트"' in color_event_page.text
    assert '/test-event`' in color_event_page.text
    assert 'edit.textContent = "수정"' in color_event_page.text
    assert 'remove.textContent = "삭제"' in color_event_page.text
    assert 'method: "DELETE"' in color_event_page.text
    assert 'id="rule-cancel"' in color_event_page.text
    assert 'id="live-mode" class="runtime-mode"' in page.text
    assert 'id="live-mode" class="runtime-mode"' in identity_page.text
    assert 'id="live-mode" class="runtime-mode"' in color_event_page.text
    assert 'health.mode === "live"' in page.text
    assert 'health.mode !== "live"' in identity_page.text
    assert 'health.mode !== "live"' in color_event_page.text
    assert "카메라 등록" in page.text
    assert "카메라 선택" in page.text
    assert 'role="tablist" aria-label="카메라 관리"' in page.text
    assert 'id="camera-register-panel" class="camera-tab-panel"' in page.text
    assert 'id="camera-select-panel" class="camera-tab-panel"' in page.text
    assert "overflow-y: auto" in page.text
    assert 'meta.textContent = camera.location || "설치 위치 미지정";' in page.text
    assert 'text("camera-meta", camera.location || "설치 위치 미지정");' in page.text
    assert "camera.rtsp_endpoint" not in page.text
    assert "camera.stream_path" not in page.text
    assert "RTSP와 HLS 연결을 항상 유지합니다" in page.text
    assert "source_on_demand: false" in page.text
    assert "camera_id: selectedCamera.id" in page.text
    assert "stream_path: selectedCamera.stream_path" in page.text
    assert 'document.addEventListener("visibilitychange"' in page.text
    assert "if (!document.hidden) reloadSelectedCamera();" in page.text
    assert 'method: "DELETE"' in page.text
    assert "세션 스냅샷과 작업 로그도 함께 삭제됩니다" in page.text
    assert 'item.rejection_reason === "no_candidates"' in page.text
    assert "후보 없음" in page.text
    assert '"rule_event_emitted"' in page.text
    assert started.status_code == 201
    assert started.json()["camera_id"] == camera.id
    assert started.json()["stream_path"] == "camera-test"
    assert started.json()["source_name"] == "camera-test"
    assert manager.last_start_kwargs["rtsp_url"].endswith("/camera-test")
    assert started.json()["sample_fps"] == 3
    assert started.json()["max_samples"] == 30
    assert sessions.json()["items"][0]["id"] == "session-1"
    assert snapshots.json()["items"][0]["sample_index"] == 1
    assert snapshots.json()["items"][0]["annotated_url"].endswith("/annotated")
    assert active_delete.status_code == 409
    assert image.content == b"jpg"
    assert stopped.status_code == 202
    assert stopped.json()["status"] == "stopped"
    assert event_stream.headers["content-type"].startswith("text/event-stream")
    assert "event: session_finished" in event_stream.text
    assert deleted.status_code == 204
    assert sessions_after_delete.json()["items"] == []


def test_dashboard_registers_identity_from_uploaded_photos(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)

    def fake_service(request) -> FaceRegistrationService:
        return FaceRegistrationService(
            IdentityRepository(request.app.state.database.path),
            FakeRegistrationExtractor(),
            photo_root=request.app.state.settings.identity_photo_dir,
        )

    monkeypatch.setattr(
        "cctv.api.test_dashboard._face_registration_service",
        fake_service,
    )
    photos = [
        {
            "file_name": f"face-{index}.jpg",
            "content_base64": base64.b64encode(f"photo-{index}".encode()).decode(),
        }
        for index in range(1, 4)
    ]

    with TestClient(create_app(settings)) as client:
        registered = client.post(
            "/test-identities",
            json={
                "display_name": "홍길동",
                "external_id": "EMP-001",
                "description": "정문 출입 테스트",
                "metadata": {"department": "보안팀"},
                "photos": photos,
            },
        )
        identities = client.get("/identities")
        identity_id = registered.json()["id"]
        embeddings = client.get(f"/identities/{identity_id}/embeddings")

    assert registered.status_code == 201
    assert registered.json()["display_name"] == "홍길동"
    assert registered.json()["photo_count"] == 3
    assert registered.json()["embedding_dimension"] == 128
    assert identities.json()["total"] == 1
    assert embeddings.json()["total"] == 3
    assert sorted(path.name for path in (settings.identity_photo_dir / identity_id).iterdir()) == [
        "photo-01.jpg",
        "photo-02.jpg",
        "photo-03.jpg",
    ]


def test_dashboard_registration_rolls_back_identity_and_photos_on_face_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)

    def failing_service(request) -> FaceRegistrationService:
        return FaceRegistrationService(
            IdentityRepository(request.app.state.database.path),
            FailingRegistrationExtractor(),
            photo_root=request.app.state.settings.identity_photo_dir,
        )

    monkeypatch.setattr(
        "cctv.api.test_dashboard._face_registration_service",
        failing_service,
    )
    photos = [
        {
            "file_name": f"face-{index}.png",
            "content_base64": base64.b64encode(f"photo-{index}".encode()).decode(),
        }
        for index in range(1, 4)
    ]

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/test-identities",
            json={"display_name": "실패 대상", "photos": photos},
        )
        identities = client.get("/identities")

    assert response.status_code == 422
    assert "exactly one face" in response.json()["detail"]
    assert identities.json()["total"] == 0
    assert list(settings.identity_photo_dir.iterdir()) == []


def test_dashboard_deletes_identity_embeddings_and_registration_photos(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        identity = client.post(
            "/identities",
            json={"display_name": "삭제 대상", "external_id": "DELETE-001"},
        ).json()
        identity_id = identity["id"]
        embedding_path = f"/identities/{identity_id}/embeddings"
        created_embedding = client.post(
            embedding_path,
            json={"model_name": "test-face", "vector": [1, 0]},
        )
        photo_directory = settings.identity_photo_dir / identity_id
        photo_directory.mkdir(parents=True)
        (photo_directory / "photo-01.jpg").write_bytes(b"registered-photo")

        deleted = client.delete(f"/test-identities/{identity_id}")
        missing_identity = client.get(f"/identities/{identity_id}")
        missing_embeddings = client.get(embedding_path)
        missing_delete = client.delete(f"/test-identities/{identity_id}")

    assert created_embedding.status_code == 201
    assert deleted.status_code == 204
    assert missing_identity.status_code == 404
    assert missing_embeddings.status_code == 404
    assert missing_delete.status_code == 404
    assert not photo_directory.exists()


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
    assert "item.annotated_url" in page.text
    assert "박스 보기" in page.text
    assert (
        'document.getElementById("face-results").replaceChildren(...replacement.children);'
        in page.text
    )


def test_dashboard_annotation_uses_one_person_id_across_changed_tracks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "sample_000001_frame_000000001_t000000500ms.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)
    settings = _settings(tmp_path)
    rendered_overlays = []

    def fake_renderer(path, overlays, *, jpeg_quality):
        assert path == snapshot
        assert jpeg_quality == settings.snapshot_jpeg_quality
        rendered_overlays.extend(overlays)
        return b"annotated-jpg"

    monkeypatch.setattr(
        "cctv.api.test_dashboard.render_snapshot_annotations",
        fake_renderer,
    )

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
        for sample_index, track_id in ((1, 7), (2, 9)):
            detection_repository.save_frame(
                run.id,
                FrameDetections(
                    source_index=sample_index,
                    sample_index=sample_index,
                    timestamp_seconds=sample_index * 0.5,
                    frame_width=640,
                    frame_height=360,
                    inference_seconds=0.01,
                    detections=(
                        Detection(
                            class_id=0,
                            label="person",
                            confidence=0.91,
                            box=BoundingBox(x1=20, y1=30, x2=220, y2=330),
                            track_id=track_id,
                        ),
                    ),
                ),
            )

        face_repository = FaceMatchRepository(settings.database_path)
        person_repository = PersonInstanceRepository(settings.database_path)
        resolver = TrackIdentityResolver(person_repository)
        for sample_index, track_id in ((1, 7), (2, 9)):
            event = face_repository.save_event(
                run.id,
                FaceMatchEventInput(
                    source_index=sample_index,
                    sample_index=sample_index,
                    source_timestamp_seconds=sample_index * 0.5,
                    face_index=0,
                    track_id=track_id,
                    face_x=70,
                    face_y=50,
                    face_width=60,
                    face_height=60,
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
            resolver.resolve(event)

        people = person_repository.get_by_tracks(run.id, {7, 9})
        manager.session = replace(manager.session, analysis_run_id=run.id)
        snapshots = client.get("/test-sessions/session-1/snapshots")
        annotated = client.get(snapshots.json()["items"][0]["annotated_url"])

    assert people[7].id == people[9].id
    assert annotated.status_code == 200
    assert annotated.content == b"annotated-jpg"
    assert annotated.headers["x-cctv-person-count"] == "1"
    assert len(rendered_overlays) == 1
    assert rendered_overlays[0].primary_label == "person 0.91 | Track 7"
    assert rendered_overlays[0].secondary_label == (
        f"Person {people[7].id[:8]} | MATCHED"
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
    assert created.json()["source_on_demand"] is False
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
        face_matches = FaceMatchRepository(settings.database_path)
        face_matches.save_event(
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
        face_matches.save_event(
            run.id,
            FaceMatchEventInput(
                source_index=20,
                sample_index=2,
                source_timestamp_seconds=1.0,
                face_index=0,
                track_id=None,
                face_x=50,
                face_y=60,
                face_width=30,
                face_height=40,
                detection_confidence=0.88,
                match_status="unknown",
                rejection_reason="no_candidates",
                identity_id=None,
                external_id=None,
                display_name=None,
                best_candidate_identity_id=None,
                best_candidate_external_id=None,
                best_similarity=0,
                second_best_similarity=None,
                similarity_threshold=0.45,
                minimum_margin=0.05,
            ),
        )

        events = client.get("/face-match-events", params={"analysis_run_id": run.id})
        summary = client.get(f"/analysis-runs/{run.id}/face-match-summary")

    assert events.status_code == 200
    assert events.json()["items"][1]["face_bounds"] == {
        "x": 10,
        "y": 20,
        "width": 30,
        "height": 40,
    }
    assert events.json()["items"][1]["external_id"] == "EMP-001"
    assert events.json()["items"][0]["match_status"] == "unknown"
    assert events.json()["items"][0]["rejection_reason"] == "no_candidates"
    assert events.json()["items"][0]["best_candidate_identity_id"] is None
    assert summary.json()["matched_faces"] == 1
    assert summary.json()["unknown_faces"] == 1


def test_dashboard_is_denied_when_feature_is_disabled(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample.jpg"
    snapshot.write_bytes(b"jpg")
    manager = FakeSessionManager(snapshot)

    with TestClient(
        create_app(_settings(tmp_path, enabled=False), test_session_manager=manager)
    ) as client:
        response = client.get("/test-dashboard")
        identity_response = client.get("/test-identities")
        color_event_response = client.get("/test-color-events")

    assert response.status_code == 403
    assert identity_response.status_code == 403
    assert color_event_response.status_code == 403


def test_dashboard_rejects_non_local_host_and_allows_local_live_mode(
    tmp_path: Path,
) -> None:
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
        live_identity_response = live_client.get("/test-identities")
        live_color_response = live_client.get("/test-color-events")

    assert remote_response.status_code == 403
    assert live_response.status_code == 200
    assert live_identity_response.status_code == 200
    assert live_color_response.status_code == 200
    assert '<span id="live-mode" class="runtime-mode" hidden>LIVE</span>' in live_response.text
    assert (
        '<span id="live-mode" class="runtime-mode" hidden>LIVE</span>'
        in live_identity_response.text
    )
    assert (
        '<span id="live-mode" class="runtime-mode" hidden>LIVE</span>'
        in live_color_response.text
    )
