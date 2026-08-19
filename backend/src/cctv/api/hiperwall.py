"""Read-only API for simulated and LIVE Hiperwall work records."""

from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from cctv.db import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DisplayActionRecord,
    DisplayActionRepository,
)

router = APIRouter(tags=["hiperwall"])


class DisplayActionResponse(BaseModel):
    id: str
    rule_event_id: str
    rule_id: str
    analysis_run_id: str
    source_name: str
    track_id: int
    event_type: str
    event_state: str
    action_type: Literal["open_source", "restore_layout"]
    mode: Literal["dry_run", "live"]
    status: Literal["simulated", "pending", "processing", "retry", "succeeded", "failed"]
    request: dict[str, Any]
    result: dict[str, Any]
    attempt_count: int
    available_at: datetime
    last_attempt_at: datetime | None
    completed_at: datetime | None
    last_error_code: str | None
    last_error_message: str | None
    created_at: datetime
    updated_at: datetime


class DisplayActionPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[DisplayActionResponse]


PageNumber = Annotated[int, Query(ge=1)]
PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


@router.get("/hiperwall-actions", response_model=DisplayActionPageResponse)
def list_hiperwall_actions(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    rule_event_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    analysis_run_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    source_name: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
    action_type: Annotated[
        Literal["open_source", "restore_layout"] | None,
        Query(),
    ] = None,
    action_status: Annotated[
        Literal["simulated", "pending", "processing", "retry", "succeeded", "failed"] | None,
        Query(alias="status"),
    ] = None,
) -> DisplayActionPageResponse:
    result = _repository(request).list(
        page=page,
        limit=limit,
        rule_event_id=rule_event_id,
        analysis_run_id=analysis_run_id,
        source_name=source_name,
        action_type=action_type,
        status=action_status,
    )
    return DisplayActionPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_response(item) for item in result.items],
    )


@router.get(
    "/hiperwall-actions/{action_id}",
    response_model=DisplayActionResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Display action not found"}},
)
def get_hiperwall_action(request: Request, action_id: str) -> DisplayActionResponse:
    record = _repository(request).get(action_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="display action not found",
        )
    return _response(record)


def _repository(request: Request) -> DisplayActionRepository:
    return DisplayActionRepository(request.app.state.database.path)


def _response(record: DisplayActionRecord) -> DisplayActionResponse:
    return DisplayActionResponse(**asdict(record))
