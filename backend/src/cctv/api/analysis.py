"""Read-only analysis-run and object-detection API."""

from dataclasses import asdict
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from cctv.db import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AnalysisRunRecord,
    AnalysisRunStatus,
    DetectionRecord,
    DetectionRepository,
    TrackObservationRecord,
    TrackRecord,
)

router = APIRouter(tags=["analysis"])


class AnalysisRunResponse(BaseModel):
    """Public analysis execution metadata and persisted aggregate counts."""

    id: str
    source_type: str
    source_name: str
    camera_id: int | None
    model_name: str
    model_sha256: str
    device: str
    input_size: int
    confidence_threshold: float
    nms_threshold: float
    sample_fps: float
    status: AnalysisRunStatus
    started_at: datetime
    completed_at: datetime | None
    processed_frames: int
    total_detections: int
    error_type: str | None


class AnalysisRunPageResponse(BaseModel):
    """Bounded page of analysis executions."""

    page: int
    limit: int
    total: int
    has_next: bool
    items: list[AnalysisRunResponse]


class BoundingBoxResponse(BaseModel):
    """Original-frame pixel coordinates for one detection."""

    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResponse(BaseModel):
    """One detection with its analysis and source-frame identity."""

    id: int
    analysis_run_id: str
    analyzed_frame_id: int
    source_index: int
    sample_index: int
    source_timestamp_seconds: float
    frame_width: int
    frame_height: int
    inference_seconds: float
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBoxResponse


class DetectionPageResponse(BaseModel):
    """Bounded page of detections matching optional filters."""

    page: int
    limit: int
    total: int
    has_next: bool
    items: list[DetectionResponse]


class TrackResponse(BaseModel):
    """One execution-scoped tracked object and its aggregate lifetime."""

    analysis_run_id: str
    track_id: int
    class_id: int
    class_name: str
    first_sample_index: int
    last_sample_index: int
    first_seen_timestamp_seconds: float
    last_seen_timestamp_seconds: float
    observation_count: int
    max_confidence: float
    is_active: bool


class TrackPageResponse(BaseModel):
    """Bounded page of tracked objects."""

    page: int
    limit: int
    total: int
    has_next: bool
    items: list[TrackResponse]


class TrackObservationResponse(BaseModel):
    """One chronological detection belonging to a tracked object."""

    id: int
    analysis_run_id: str
    track_id: int
    analyzed_frame_id: int
    detection_id: int
    source_index: int
    sample_index: int
    source_timestamp_seconds: float
    confidence: float
    box: BoundingBoxResponse


class TrackObservationPageResponse(BaseModel):
    """Bounded chronological observation page for one track."""

    page: int
    limit: int
    total: int
    has_next: bool
    items: list[TrackObservationResponse]


PageNumber = Annotated[int, Query(ge=1)]
PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


@router.get("/analysis-runs", response_model=AnalysisRunPageResponse)
def list_analysis_runs(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    run_status: Annotated[AnalysisRunStatus | None, Query(alias="status")] = None,
) -> AnalysisRunPageResponse:
    """List recent analysis executions with optional status filtering."""
    result = _repository(request).list_analysis_runs(
        page=page,
        limit=limit,
        status=run_status,
    )
    return AnalysisRunPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_analysis_run_response(item) for item in result.items],
    )


@router.get(
    "/analysis-runs/{analysis_run_id}",
    response_model=AnalysisRunResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Analysis run not found"}},
)
def get_analysis_run(request: Request, analysis_run_id: str) -> AnalysisRunResponse:
    """Return one execution without exposing source or model filesystem paths."""
    result = _repository(request).get_analysis_run(analysis_run_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="analysis run not found")
    return _analysis_run_response(result)


@router.get("/detections", response_model=DetectionPageResponse)
def list_detections(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    analysis_run_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    track_id: Annotated[int | None, Query(ge=1)] = None,
    class_name: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> DetectionPageResponse:
    """Query detections by execution, per-run track, class, and confidence."""
    if track_id is not None and analysis_run_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="analysis_run_id is required when filtering by track_id",
        )
    result = _repository(request).list_detections(
        page=page,
        limit=limit,
        analysis_run_id=analysis_run_id,
        track_id=track_id,
        class_name=class_name,
        min_confidence=min_confidence,
    )
    return DetectionPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_detection_response(item) for item in result.items],
    )


@router.get("/tracks", response_model=TrackPageResponse)
def list_tracks(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    analysis_run_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    class_name: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    min_observations: Annotated[int | None, Query(ge=1)] = None,
    observed_from_seconds: Annotated[float | None, Query(ge=0)] = None,
    observed_to_seconds: Annotated[float | None, Query(ge=0)] = None,
    active: Annotated[bool | None, Query()] = None,
) -> TrackPageResponse:
    """Query tracks by class, source-time overlap, lifetime length, and active state."""
    if (
        observed_from_seconds is not None
        and observed_to_seconds is not None
        and observed_from_seconds > observed_to_seconds
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="observed_from_seconds must be less than or equal to observed_to_seconds",
        )
    result = _repository(request).list_tracks(
        page=page,
        limit=limit,
        analysis_run_id=analysis_run_id,
        class_name=class_name,
        min_observations=min_observations,
        observed_from_seconds=observed_from_seconds,
        observed_to_seconds=observed_to_seconds,
        is_active=active,
    )
    return TrackPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_track_response(item) for item in result.items],
    )


@router.get(
    "/analysis-runs/{analysis_run_id}/tracks/{track_id}",
    response_model=TrackResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Track not found"}},
)
def get_track(request: Request, analysis_run_id: str, track_id: int) -> TrackResponse:
    """Return one execution-scoped tracked object."""
    if track_id < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="track_id must be at least 1",
        )
    result = _repository(request).get_track(analysis_run_id, track_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="track not found")
    return _track_response(result)


@router.get(
    "/analysis-runs/{analysis_run_id}/tracks/{track_id}/observations",
    response_model=TrackObservationPageResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Track not found"}},
)
def list_track_observations(
    request: Request,
    analysis_run_id: str,
    track_id: int,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
) -> TrackObservationPageResponse:
    """Return one track's detection history in source-frame order."""
    repository = _repository(request)
    if track_id < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="track_id must be at least 1",
        )
    if repository.get_track(analysis_run_id, track_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="track not found")
    result = repository.list_track_observations(
        analysis_run_id=analysis_run_id,
        track_id=track_id,
        page=page,
        limit=limit,
    )
    return TrackObservationPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_track_observation_response(item) for item in result.items],
    )


def _repository(request: Request) -> DetectionRepository:
    return DetectionRepository(request.app.state.database.path)


def _analysis_run_response(record: AnalysisRunRecord) -> AnalysisRunResponse:
    return AnalysisRunResponse(**asdict(record))


def _detection_response(record: DetectionRecord) -> DetectionResponse:
    return DetectionResponse(
        id=record.id,
        analysis_run_id=record.analysis_run_id,
        analyzed_frame_id=record.analyzed_frame_id,
        source_index=record.source_index,
        sample_index=record.sample_index,
        source_timestamp_seconds=record.source_timestamp_seconds,
        frame_width=record.frame_width,
        frame_height=record.frame_height,
        inference_seconds=record.inference_seconds,
        track_id=record.track_id,
        class_id=record.class_id,
        class_name=record.class_name,
        confidence=record.confidence,
        box=BoundingBoxResponse(
            x1=record.x1,
            y1=record.y1,
            x2=record.x2,
            y2=record.y2,
        ),
    )


def _track_response(record: TrackRecord) -> TrackResponse:
    return TrackResponse(**asdict(record))


def _track_observation_response(record: TrackObservationRecord) -> TrackObservationResponse:
    return TrackObservationResponse(
        id=record.id,
        analysis_run_id=record.analysis_run_id,
        track_id=record.track_id,
        analyzed_frame_id=record.analyzed_frame_id,
        detection_id=record.detection_id,
        source_index=record.source_index,
        sample_index=record.sample_index,
        source_timestamp_seconds=record.source_timestamp_seconds,
        confidence=record.confidence,
        box=BoundingBoxResponse(x1=record.x1, y1=record.y1, x2=record.x2, y2=record.y2),
    )
