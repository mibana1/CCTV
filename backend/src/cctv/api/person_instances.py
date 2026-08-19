"""Read-only APIs for session-scoped people and their linked tracker IDs."""

from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from cctv.db import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, PersonInstanceRepository

router = APIRouter(tags=["person-instances"])


class PersonInstanceResponse(BaseModel):
    id: str
    analysis_run_id: str
    status: Literal["matched", "unknown"]
    identity_id: str | None
    external_id: str | None
    display_name: str | None
    best_candidate_identity_id: str
    best_candidate_external_id: str | None
    best_similarity: float
    first_sample_index: int
    last_sample_index: int
    first_seen_timestamp_seconds: float
    last_seen_timestamp_seconds: float
    last_face_x: int
    last_face_y: int
    last_face_width: int
    last_face_height: int
    track_count: int
    created_at: datetime
    updated_at: datetime


class PersonInstancePageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[PersonInstanceResponse]


class TrackIdentityLinkResponse(BaseModel):
    id: int
    analysis_run_id: str
    track_id: int
    person_instance_id: str
    linked_by: Literal[
        "face_match",
        "identity_match",
        "candidate_stitch",
        "unknown_observation",
    ]
    confidence: float
    first_sample_index: int
    last_sample_index: int
    first_seen_timestamp_seconds: float
    last_seen_timestamp_seconds: float
    created_at: datetime
    updated_at: datetime


class TrackIdentityLinkPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[TrackIdentityLinkResponse]


class PersonInstanceRunSummaryResponse(BaseModel):
    analysis_run_id: str
    total_instances: int
    matched_instances: int
    unknown_instances: int
    linked_tracks: int


PageNumber = Annotated[int, Query(ge=1)]
PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


@router.get("/person-instances", response_model=PersonInstancePageResponse)
def list_person_instances(
    request: Request,
    analysis_run_id: Annotated[str, Query(min_length=1, max_length=128)],
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    instance_status: Annotated[
        Literal["matched", "unknown"] | None,
        Query(alias="status"),
    ] = None,
    identity_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> PersonInstancePageResponse:
    result = _repository(request).list_instances(
        analysis_run_id=analysis_run_id,
        page=page,
        limit=limit,
        status=instance_status,
        identity_id=identity_id,
    )
    return PersonInstancePageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[PersonInstanceResponse(**asdict(item)) for item in result.items],
    )


@router.get(
    "/analysis-runs/{analysis_run_id}/person-instances/{person_instance_id}/tracks",
    response_model=TrackIdentityLinkPageResponse,
)
def list_person_instance_tracks(
    request: Request,
    analysis_run_id: str,
    person_instance_id: str,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
) -> TrackIdentityLinkPageResponse:
    repository = _repository(request)
    instance = repository.get_instance(person_instance_id)
    if instance is None or instance.analysis_run_id != analysis_run_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="person instance not found",
        )
    result = repository.list_links(person_instance_id, page=page, limit=limit)
    return TrackIdentityLinkPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[TrackIdentityLinkResponse(**asdict(item)) for item in result.items],
    )


@router.get(
    "/analysis-runs/{analysis_run_id}/person-instance-summary",
    response_model=PersonInstanceRunSummaryResponse,
)
def get_person_instance_summary(
    request: Request,
    analysis_run_id: str,
) -> PersonInstanceRunSummaryResponse:
    summary = _repository(request).summarize_run(analysis_run_id)
    return PersonInstanceRunSummaryResponse(**asdict(summary))


def _repository(request: Request) -> PersonInstanceRepository:
    return PersonInstanceRepository(request.app.state.database.path)


__all__ = ["router"]

