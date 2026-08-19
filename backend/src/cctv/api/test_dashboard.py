"""Local dry-run RTSP test control, event, and dashboard endpoints."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from cctv.cameras import CameraCredentialError, CameraManager, CameraProvisioningError
from cctv.core.settings import AppMode
from cctv.dashboard import (
    SessionConflictError,
    SessionNotFoundError,
    TestSessionManager,
    TestSessionSnapshot,
    TestSessionStatus,
)
from cctv.dashboard.sessions import TERMINAL_SESSION_STATUSES, event_to_dict
from cctv.db import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    CameraProvisioningStatus,
    CameraRecord,
    CameraRepository,
    DetectionRepository,
    FaceMatchEventRecord,
    FaceMatchRepository,
    IdentityRepository,
    PersonInstanceRepository,
)
from cctv.identity import (
    MAX_REGISTRATION_PHOTOS,
    MIN_REGISTRATION_PHOTOS,
    SUPPORTED_PHOTO_SUFFIXES,
    FaceEmbeddingError,
    FaceModelLoadError,
    FaceRegistrationError,
    FaceRegistrationService,
    OpenCvSFaceExtractor,
)
from cctv.media import (
    SnapshotEncodingError,
    SnapshotOverlay,
    render_snapshot_annotations,
    sample_index_from_snapshot_name,
)

router = APIRouter(tags=["test-dashboard"])
_DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "page.html"
_PEOPLE_PAGE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "people.html"
_MAX_REGISTRATION_PHOTO_BYTES = 8 * 1024 * 1024
_MAX_REGISTRATION_PHOTO_BASE64_LENGTH = 11_200_000


class TestSessionStartRequest(BaseModel):
    """Bounded options accepted by the local validation runner."""

    camera_id: int = Field(gt=0)
    stream_path: str = Field(min_length=1, max_length=128)
    sample_fps: float = Field(default=2, gt=0, le=30)
    max_samples: int = Field(default=240, ge=1, le=7_200)
    save_snapshots: bool = True
    match_faces: bool = True


class TestSessionResponse(BaseModel):
    id: str
    status: TestSessionStatus
    camera_id: int
    stream_path: str
    source_name: str
    sample_fps: float
    max_samples: int
    save_snapshots: bool
    match_faces: bool
    analysis_run_id: str | None
    processed_samples: int
    connection_count: int
    reconnect_count: int
    elapsed_seconds: float
    cpu_percent: float | None
    memory_mib: float | None
    peak_cpu_percent: float | None
    peak_memory_mib: float | None
    snapshot_count: int
    started_at: datetime
    completed_at: datetime | None
    last_message: str | None
    error_type: str | None
    result_summary: dict[str, Any] | None


class TestSessionListResponse(BaseModel):
    items: list[TestSessionResponse]


class SnapshotResponse(BaseModel):
    name: str
    sample_index: int | None
    size_bytes: int
    url: str
    annotated_url: str


class SnapshotListResponse(BaseModel):
    items: list[SnapshotResponse]


class FaceBoundsResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceMatchEventResponse(BaseModel):
    id: int
    analysis_run_id: str
    source_index: int
    sample_index: int
    source_timestamp_seconds: float
    face_index: int
    track_id: int | None
    person_instance_id: str | None
    face_bounds: FaceBoundsResponse
    detection_confidence: float
    match_status: Literal["matched", "unknown"]
    rejection_reason: str | None
    identity_id: str | None
    external_id: str | None
    display_name: str | None
    best_candidate_identity_id: str | None
    best_candidate_external_id: str | None
    best_similarity: float
    second_best_similarity: float | None
    similarity_threshold: float
    minimum_margin: float
    created_at: datetime


class FaceMatchEventPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[FaceMatchEventResponse]


class FaceMatchRunSummaryResponse(BaseModel):
    analysis_run_id: str
    detected_faces: int
    matched_faces: int
    unknown_faces: int
    best_similarity: float | None


class CameraCreateRequest(BaseModel):
    """RTSP camera details used to create a unique MediaMTX path."""

    name: str = Field(min_length=1, max_length=100)
    location: str | None = Field(default=None, max_length=200)
    rtsp_url: str = Field(min_length=1, max_length=2_048)
    username: SecretStr | None = Field(default=None, max_length=256)
    password: SecretStr | None = Field(default=None, max_length=512)
    source_on_demand: bool = False


class CameraUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    location: str | None = Field(default=None, max_length=200)
    enabled: bool | None = None
    rtsp_url: str | None = Field(default=None, min_length=1, max_length=2_048)
    username: SecretStr | None = Field(default=None, max_length=256)
    password: SecretStr | None = Field(default=None, max_length=512)
    source_on_demand: bool | None = None


class CameraResponse(BaseModel):
    id: int
    name: str
    location: str | None
    stream_path: str
    enabled: bool
    rtsp_endpoint: str | None
    credentials_configured: bool
    source_on_demand: bool
    provisioning_status: str
    last_sync_error: str | None
    last_synced_at: datetime | None
    preview_url: str
    created_at: datetime
    updated_at: datetime


class CameraPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[CameraResponse]


class CameraLiveStatusResponse(BaseModel):
    camera_id: int
    configured: bool
    online: bool
    available: bool


class RegistrationPhotoRequest(BaseModel):
    file_name: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=4, max_length=_MAX_REGISTRATION_PHOTO_BASE64_LENGTH)


class TestIdentityRegistrationRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)
    external_id: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, min_length=1, max_length=1_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    photos: list[RegistrationPhotoRequest] = Field(
        min_length=MIN_REGISTRATION_PHOTOS,
        max_length=MAX_REGISTRATION_PHOTOS,
    )


class RegisteredPhotoResponse(BaseModel):
    photo_name: str
    detection_confidence: float
    embedding_id: str


class TestIdentityRegistrationResponse(BaseModel):
    id: str
    external_id: str | None
    display_name: str
    description: str | None
    metadata: dict[str, Any]
    enabled: bool
    photo_count: int
    model_name: str
    model_version: str
    embedding_dimension: int
    photos: list[RegisteredPhotoResponse]


def require_test_dashboard_access(request: Request) -> None:
    """Limit process-control and biometric test views to local dry-run development."""
    settings = request.app.state.settings
    local_environment = settings.app_env.strip().casefold() in {"development", "local", "test"}
    local_host = request.url.hostname in {"127.0.0.1", "localhost", "::1", "testserver"}
    if (
        not settings.test_dashboard_enabled
        or settings.app_mode is not AppMode.DRY_RUN
        or not local_environment
        or not local_host
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="test dashboard is available only from localhost in dry-run development mode",
        )


TestAccess = Annotated[None, Depends(require_test_dashboard_access)]


@router.get("/test-dashboard", response_class=HTMLResponse)
def dashboard_page(_: TestAccess) -> HTMLResponse:
    """Serve the dependency-free dashboard from the API origin."""
    return HTMLResponse(_DASHBOARD_PATH.read_text(encoding="utf-8"))


@router.get("/test-identities", response_class=HTMLResponse)
def identity_registration_page(_: TestAccess) -> HTMLResponse:
    """Serve the local person registration page from the API origin."""
    return HTMLResponse(_PEOPLE_PAGE_PATH.read_text(encoding="utf-8"))


@router.post(
    "/test-identities",
    response_model=TestIdentityRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "External ID already registered or face models unavailable"
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid person details or registration photos"
        },
    },
)
def register_test_identity(
    request: Request,
    payload: TestIdentityRegistrationRequest,
    _: TestAccess,
) -> TestIdentityRegistrationResponse:
    """Register one local test identity from three to five browser-uploaded photos."""
    settings = request.app.state.settings
    repository = IdentityRepository(request.app.state.database.path)
    photo_root = settings.identity_photo_dir.expanduser().resolve()
    uploaded_directory: Path | None = None
    identity_id: str | None = None
    service: FaceRegistrationService | None = None
    result = None
    try:
        uploaded_directory = _write_registration_photos(photo_root, payload.photos)
        service = _face_registration_service(request)
        identity = repository.create_identity(
            display_name=payload.display_name,
            external_id=payload.external_id,
            description=payload.description,
            metadata=payload.metadata,
        )
        identity_id = identity.id
        identity_directory = photo_root / identity.id
        uploaded_directory.replace(identity_directory)
        uploaded_directory = identity_directory
        result = service.register(identity.id, photo_directory=identity_directory)
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="external_id is already registered",
        ) from error
    except FaceModelLoadError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"face recognition model is unavailable: {error}",
        ) from error
    except (FaceRegistrationError, FaceEmbeddingError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    finally:
        if result is None:
            if identity_id is not None:
                repository.delete_identity(identity_id)
            if uploaded_directory is not None:
                shutil.rmtree(uploaded_directory, ignore_errors=True)

    assert service is not None
    metadata = service.extractor.metadata
    return TestIdentityRegistrationResponse(
        id=result.identity.id,
        external_id=result.identity.external_id,
        display_name=result.identity.display_name,
        description=result.identity.description,
        metadata=result.identity.metadata,
        enabled=result.identity.enabled,
        photo_count=len(result.items),
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        embedding_dimension=metadata.dimension,
        photos=[
            RegisteredPhotoResponse(
                photo_name=item.photo_name,
                detection_confidence=item.detection_confidence,
                embedding_id=item.embedding.id,
            )
            for item in result.items
        ],
    )


@router.delete(
    "/test-identities/{identity_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Identity not found"}},
)
def delete_test_identity(
    request: Request,
    identity_id: str,
    _: TestAccess,
) -> Response:
    """Delete one test identity, its embeddings, and its uploaded photo directory."""
    repository = IdentityRepository(request.app.state.database.path)
    identity = repository.get_identity(identity_id)
    if identity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="identity not found")

    photo_root = request.app.state.settings.identity_photo_dir.expanduser().resolve()
    photo_directory = photo_root / identity.id
    if photo_directory.parent != photo_root:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="identity photo directory is outside the configured photo root",
        )
    if not repository.delete_identity(identity.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="identity not found")
    if photo_directory.is_symlink():
        photo_directory.unlink(missing_ok=True)
    elif photo_directory.is_dir():
        shutil.rmtree(photo_directory)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/test-cameras",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_test_camera(
    request: Request,
    payload: CameraCreateRequest,
    _: TestAccess,
) -> CameraResponse:
    try:
        manager = _camera_manager(request)
        record = manager.register_camera(
            name=payload.name,
            location=payload.location,
            rtsp_url=payload.rtsp_url,
            username=_secret_value(payload.username),
            password=_secret_value(payload.password),
            source_on_demand=payload.source_on_demand,
        )
    except (CameraCredentialError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except CameraProvisioningError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{error} 등록 ID: {error.camera_id}",
        ) from error
    return _camera_response(request, record)


@router.get("/test-cameras", response_model=CameraPageResponse)
def list_test_cameras(
    request: Request,
    _: TestAccess,
    page: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    enabled: Annotated[bool | None, Query()] = None,
) -> CameraPageResponse:
    result = _camera_repository(request).list_cameras(
        page=page,
        limit=limit,
        enabled=enabled,
    )
    return CameraPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_camera_response(request, item) for item in result.items],
    )


@router.get(
    "/test-cameras/{camera_id}",
    response_model=CameraResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Camera not found"}},
)
def get_test_camera(request: Request, camera_id: int, _: TestAccess) -> CameraResponse:
    record = _camera_repository(request).get_camera(camera_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    return _camera_response(request, record)


@router.patch(
    "/test-cameras/{camera_id}",
    response_model=CameraResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Camera not found"}},
)
def update_test_camera(
    request: Request,
    camera_id: int,
    payload: CameraUpdateRequest,
    _: TestAccess,
) -> CameraResponse:
    supplied = payload.model_fields_set
    if not supplied:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="at least one camera field is required",
        )
    changes = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if key in {"name", "location", "enabled"}
    }
    try:
        record = _camera_manager(request).update_camera(
            camera_id,
            changes,
            rtsp_url=payload.rtsp_url if "rtsp_url" in supplied else None,
            username=(
                _secret_value(payload.username) if "username" in supplied else None
            ),
            password=(
                _secret_value(payload.password) if "password" in supplied else None
            ),
            source_on_demand=(
                payload.source_on_demand if "source_on_demand" in supplied else None
            ),
        )
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found") from error
    except (CameraCredentialError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except CameraProvisioningError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    return _camera_response(request, record)


@router.delete(
    "/test-cameras/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_test_camera(request: Request, camera_id: int, _: TestAccess) -> Response:
    try:
        deleted = _camera_manager(request).delete_camera(camera_id)
    except CameraProvisioningError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/test-cameras/{camera_id}/sync", response_model=CameraResponse)
def sync_test_camera(request: Request, camera_id: int, _: TestAccess) -> CameraResponse:
    try:
        record = _camera_manager(request).sync_camera(camera_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found") from error
    except CameraProvisioningError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    return _camera_response(request, record)


@router.get(
    "/test-cameras/{camera_id}/status",
    response_model=CameraLiveStatusResponse,
)
def get_test_camera_status(
    request: Request,
    camera_id: int,
    _: TestAccess,
) -> CameraLiveStatusResponse:
    try:
        result = _camera_manager(request).get_live_status(camera_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found") from error
    except CameraProvisioningError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    return CameraLiveStatusResponse(
        camera_id=camera_id,
        configured=result is not None,
        online=result.online if result is not None else False,
        available=result.available if result is not None else False,
    )


@router.post(
    "/test-sessions",
    response_model=TestSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_test_session(
    request: Request,
    payload: TestSessionStartRequest,
    _: TestAccess,
) -> TestSessionResponse:
    manager = _manager(request)
    camera = _camera_repository(request).get_camera(payload.camera_id)
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    if camera.stream_path != payload.stream_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="선택한 카메라의 스트림 경로가 등록 정보와 일치하지 않습니다.",
        )
    if not camera.enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="선택한 카메라가 비활성화되어 있습니다.",
        )
    if camera.provisioning_status not in {
        CameraProvisioningStatus.ACTIVE,
        CameraProvisioningStatus.EXTERNAL,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="선택한 카메라 스트림이 아직 준비되지 않았습니다.",
        )
    try:
        session = manager.start(
            camera_id=camera.id,
            stream_path=camera.stream_path,
            rtsp_url=_camera_rtsp_worker_url(
                request.app.state.settings.rtsp_worker_url,
                camera.stream_path,
            ),
            sample_fps=payload.sample_fps,
            max_samples=payload.max_samples,
            save_snapshots=payload.save_snapshots,
            match_faces=payload.match_faces,
        )
    except SessionConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"required model file is missing: {error}",
        ) from error
    return _session_response(session)


@router.get("/test-sessions", response_model=TestSessionListResponse)
def list_test_sessions(request: Request, _: TestAccess) -> TestSessionListResponse:
    return TestSessionListResponse(
        items=[_session_response(item) for item in _manager(request).list()]
    )


@router.get(
    "/test-sessions/{session_id}",
    response_model=TestSessionResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Test session not found"}},
)
def get_test_session(request: Request, session_id: str, _: TestAccess) -> TestSessionResponse:
    try:
        return _session_response(_manager(request).get(session_id))
    except SessionNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found") from error


@router.post(
    "/test-sessions/{session_id}/stop",
    response_model=TestSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def stop_test_session(request: Request, session_id: str, _: TestAccess) -> TestSessionResponse:
    try:
        return _session_response(_manager(request).stop(session_id))
    except SessionNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found") from error


@router.delete("/test-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_test_session(request: Request, session_id: str, _: TestAccess) -> Response:
    try:
        _manager(request).delete(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found") from error
    except SessionConflictError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/test-sessions/{session_id}/events")
async def stream_test_session_events(
    request: Request,
    session_id: str,
    _: TestAccess,
    after: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    manager = _manager(request)
    try:
        manager.get(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found") from error

    async def event_stream():
        cursor = after
        idle_cycles = 0
        while True:
            if await request.is_disconnected():
                return
            events = manager.events(session_id, after=cursor)
            for event in events:
                cursor = event.sequence
                payload = json.dumps(event_to_dict(event), ensure_ascii=False, default=str)
                yield f"id: {event.sequence}\nevent: {event.event}\ndata: {payload}\n\n"
            session = manager.get(session_id)
            if session.status in TERMINAL_SESSION_STATUSES and not events:
                return
            idle_cycles += 1
            if idle_cycles % 20 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/test-sessions/{session_id}/snapshots",
    response_model=SnapshotListResponse,
)
def list_test_snapshots(
    request: Request,
    session_id: str,
    _: TestAccess,
) -> SnapshotListResponse:
    manager = _manager(request)
    try:
        snapshots = manager.snapshots(session_id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found") from error
    return SnapshotListResponse(
        items=[
            SnapshotResponse(
                **asdict(item),
                url=f"/test-sessions/{session_id}/snapshots/{item.name}",
                annotated_url=(
                    f"/test-sessions/{session_id}/snapshots/{item.name}/annotated"
                ),
            )
            for item in snapshots
        ]
    )


@router.get("/test-sessions/{session_id}/snapshots/{name}", response_class=FileResponse)
def get_test_snapshot(
    request: Request,
    session_id: str,
    name: str,
    _: TestAccess,
) -> FileResponse:
    try:
        path = _manager(request).snapshot_path(session_id, name)
    except (SessionNotFoundError, FileNotFoundError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="snapshot not found") from error
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get(
    "/test-sessions/{session_id}/snapshots/{name}/annotated",
    response_class=Response,
)
def get_annotated_test_snapshot(
    request: Request,
    session_id: str,
    name: str,
    _: TestAccess,
) -> Response:
    """Render a source snapshot using the latest track-to-person associations."""
    manager = _manager(request)
    try:
        session = manager.get(session_id)
        path = manager.snapshot_path(session_id, name)
    except (SessionNotFoundError, FileNotFoundError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="snapshot not found") from error

    sample_index = sample_index_from_snapshot_name(name)
    if session.analysis_run_id is None or sample_index is None:
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    database_path = request.app.state.database.path
    detections = DetectionRepository(database_path).list_detections(
        analysis_run_id=session.analysis_run_id,
        sample_index=sample_index,
        page=1,
        limit=MAX_PAGE_SIZE,
    )
    tracked_class_names = request.app.state.settings.tracker_class_name_set
    tracked_detections = tuple(
        item
        for item in detections.items
        if item.track_id is not None
        and item.class_name.strip().casefold() in tracked_class_names
    )
    track_ids = {
        item.track_id for item in tracked_detections if item.track_id is not None
    }
    people_by_track = PersonInstanceRepository(database_path).get_by_tracks(
        session.analysis_run_id,
        track_ids,
    )
    overlays = tuple(
        _snapshot_overlay(item, people_by_track.get(item.track_id))
        for item in tracked_detections
    )
    try:
        content = render_snapshot_annotations(
            path,
            overlays,
            jpeg_quality=request.app.state.settings.snapshot_jpeg_quality,
        )
    except SnapshotEncodingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-CCTV-Detection-Count": str(len(overlays)),
            "X-CCTV-Person-Count": str(len(people_by_track)),
        },
    )


@router.get("/face-match-events", response_model=FaceMatchEventPageResponse)
def list_face_match_events(
    request: Request,
    _: TestAccess,
    analysis_run_id: Annotated[str, Query(min_length=1, max_length=128)],
    page: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    match_status: Annotated[Literal["matched", "unknown"] | None, Query()] = None,
) -> FaceMatchEventPageResponse:
    result = _face_repository(request).list_events(
        analysis_run_id=analysis_run_id,
        page=page,
        limit=limit,
        match_status=match_status,
    )
    return FaceMatchEventPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_face_event_response(item) for item in result.items],
    )


@router.get(
    "/analysis-runs/{analysis_run_id}/face-match-summary",
    response_model=FaceMatchRunSummaryResponse,
)
def get_face_match_summary(
    request: Request,
    analysis_run_id: str,
    _: TestAccess,
) -> FaceMatchRunSummaryResponse:
    result = _face_repository(request).summarize_run(analysis_run_id)
    return FaceMatchRunSummaryResponse(**asdict(result))


def _write_registration_photos(
    photo_root: Path,
    photos: list[RegistrationPhotoRequest],
) -> Path:
    photo_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".upload-", dir=photo_root))
    try:
        for index, photo in enumerate(photos, start=1):
            suffix = Path(photo.file_name).suffix.casefold()
            if suffix not in SUPPORTED_PHOTO_SUFFIXES:
                raise FaceRegistrationError(
                    f"registration photo must be JPG or PNG: {photo.file_name}"
                )
            try:
                content = base64.b64decode(photo.content_base64, validate=True)
            except (binascii.Error, ValueError) as error:
                raise FaceRegistrationError(
                    f"registration photo is not valid base64: {photo.file_name}"
                ) from error
            if not content:
                raise FaceRegistrationError(
                    f"registration photo is empty: {photo.file_name}"
                )
            if len(content) > _MAX_REGISTRATION_PHOTO_BYTES:
                raise FaceRegistrationError(
                    f"registration photo exceeds 8 MiB: {photo.file_name}"
                )
            (directory / f"photo-{index:02d}{suffix}").write_bytes(content)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return directory


def _face_registration_service(request: Request) -> FaceRegistrationService:
    settings = request.app.state.settings
    extractor = OpenCvSFaceExtractor(
        settings.face_detection_model_path,
        settings.face_embedding_model_path,
        score_threshold=settings.face_detection_score_threshold,
        nms_threshold=settings.face_detection_nms_threshold,
        top_k=settings.face_detection_top_k,
        max_input_dimension=settings.face_detection_max_input_dimension,
    )
    return FaceRegistrationService(
        IdentityRepository(request.app.state.database.path),
        extractor,
        photo_root=settings.identity_photo_dir,
    )


def _manager(request: Request) -> TestSessionManager:
    return request.app.state.test_session_manager


def _face_repository(request: Request) -> FaceMatchRepository:
    return FaceMatchRepository(request.app.state.database.path)


def _camera_repository(request: Request) -> CameraRepository:
    return CameraRepository(request.app.state.database.path)


def _camera_manager(request: Request) -> CameraManager:
    manager = request.app.state.camera_manager
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="dynamic MediaMTX camera registration is disabled",
        )
    return manager


def _secret_value(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None else None


def _camera_response(request: Request, record: CameraRecord) -> CameraResponse:
    encoded_path = quote(record.stream_path, safe="/")
    preview_url = f"{request.app.state.settings.media_hls_base_url}/{encoded_path}"
    return CameraResponse(**asdict(record), preview_url=preview_url)


def _camera_rtsp_worker_url(configured_url: str, stream_path: str) -> str:
    parsed = urlsplit(configured_url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/{quote(stream_path, safe='/')}",
            "",
            "",
        )
    )


def _session_response(session: TestSessionSnapshot) -> TestSessionResponse:
    return TestSessionResponse(**asdict(session))


def _face_event_response(record: FaceMatchEventRecord) -> FaceMatchEventResponse:
    values = asdict(record)
    for key in ("face_x", "face_y", "face_width", "face_height"):
        values.pop(key)
    return FaceMatchEventResponse(
        **values,
        face_bounds=FaceBoundsResponse(
            x=record.face_x,
            y=record.face_y,
            width=record.face_width,
            height=record.face_height,
        ),
    )


def _snapshot_overlay(record, person) -> SnapshotOverlay:
    primary_label = f"{record.class_name} {record.confidence:.2f}"
    if record.track_id is not None:
        primary_label += f" | Track {record.track_id}"
    secondary_label = None
    color = (150, 150, 150)
    if person is not None:
        secondary_label = f"Person {person.id[:8]} | {person.status.upper()}"
        color = (42, 190, 80) if person.status == "matched" else (0, 184, 255)
    elif record.track_id is not None:
        color = (255, 140, 0)
    return SnapshotOverlay(
        x1=record.x1,
        y1=record.y1,
        x2=record.x2,
        y2=record.y2,
        primary_label=primary_label,
        secondary_label=secondary_label,
        color=color,
    )


__all__ = ["require_test_dashboard_access", "router"]
