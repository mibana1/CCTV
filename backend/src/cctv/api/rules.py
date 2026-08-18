"""Rule configuration and emitted-event API."""

import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from cctv.db import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    RuleEventRecord,
    RuleRecord,
    RuleRepository,
)
from cctv.rules import RuleConfigurationError, RuleDefinition, RuleEngine

router = APIRouter(tags=["rules"])


class RuleCreateRequest(BaseModel):
    """Evaluator-neutral rule configuration using normalized 0..1 coordinates."""

    name: str = Field(min_length=1, max_length=128)
    source_name: str = Field(min_length=1, max_length=255)
    rule_type: str = Field(min_length=1, max_length=64)
    class_name: str | None = Field(default=None, min_length=1, max_length=128)
    geometry: dict[str, Any]
    parameters: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RuleEnabledRequest(BaseModel):
    enabled: bool


class RuleResponse(BaseModel):
    id: str
    name: str
    source_name: str
    rule_type: str
    class_name: str | None
    geometry: dict[str, Any]
    parameters: dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime


class RulePageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[RuleResponse]


class RuleTypeListResponse(BaseModel):
    items: list[str]


class RuleEventResponse(BaseModel):
    id: str
    rule_id: str
    rule_name: str
    rule_type: str
    analysis_run_id: str
    track_id: int
    analyzed_frame_id: int
    event_type: str
    event_state: str
    occurred_at_seconds: float
    class_name: str
    confidence: float
    payload: dict[str, Any]
    created_at: datetime


class RuleEventPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[RuleEventResponse]


PageNumber = Annotated[int, Query(ge=1)]
PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


@router.get("/rule-types", response_model=RuleTypeListResponse)
def list_rule_types() -> RuleTypeListResponse:
    """List evaluator types registered by this application build."""
    return RuleTypeListResponse(items=list(RuleEngine(()).supported_rule_types))


@router.post(
    "/rules",
    response_model=RuleResponse,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "Rule name already exists for source"}},
)
def create_rule(request: Request, body: RuleCreateRequest) -> RuleResponse:
    """Validate and persist one rule; active workers load it on their next start."""
    definition = RuleDefinition(
        id="pending",
        name=body.name,
        source_name=body.source_name,
        rule_type=body.rule_type,
        class_name=body.class_name,
        geometry=body.geometry,
        parameters=body.parameters,
        enabled=True,
    )
    try:
        RuleEngine((definition,))
    except RuleConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    try:
        record = _repository(request).create_rule(
            name=body.name,
            source_name=body.source_name,
            rule_type=body.rule_type,
            class_name=body.class_name,
            geometry=body.geometry,
            parameters=body.parameters,
            enabled=body.enabled,
        )
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a rule with this name already exists for the source",
        ) from error
    return _rule_response(record)


@router.get("/rules", response_model=RulePageResponse)
def list_rules(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    source_name: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
    rule_type: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    enabled: Annotated[bool | None, Query()] = None,
) -> RulePageResponse:
    result = _repository(request).list_rules(
        page=page,
        limit=limit,
        source_name=source_name,
        rule_type=rule_type,
        enabled=enabled,
    )
    return RulePageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_rule_response(item) for item in result.items],
    )


@router.get(
    "/rules/{rule_id}",
    response_model=RuleResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Rule not found"}},
)
def get_rule(request: Request, rule_id: str) -> RuleResponse:
    record = _repository(request).get_rule(rule_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="rule not found")
    return _rule_response(record)


@router.patch(
    "/rules/{rule_id}/enabled",
    response_model=RuleResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Rule not found"}},
)
def set_rule_enabled(
    request: Request,
    rule_id: str,
    body: RuleEnabledRequest,
) -> RuleResponse:
    """Toggle a rule; running workers retain their startup snapshot."""
    repository = _repository(request)
    current = repository.get_rule(rule_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="rule not found")
    if body.enabled:
        try:
            RuleEngine((current.to_definition(),))
        except RuleConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
    try:
        record = repository.set_rule_enabled(rule_id, body.enabled)
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="rule not found"
        ) from error
    return _rule_response(record)


@router.get("/rule-events", response_model=RuleEventPageResponse)
def list_rule_events(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    rule_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    analysis_run_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    track_id: Annotated[int | None, Query(ge=1)] = None,
    event_type: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    event_state: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    occurred_from_seconds: Annotated[float | None, Query(ge=0)] = None,
    occurred_to_seconds: Annotated[float | None, Query(ge=0)] = None,
) -> RuleEventPageResponse:
    if (
        occurred_from_seconds is not None
        and occurred_to_seconds is not None
        and occurred_from_seconds > occurred_to_seconds
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="occurred_from_seconds must be less than or equal to occurred_to_seconds",
        )
    result = _repository(request).list_events(
        page=page,
        limit=limit,
        rule_id=rule_id,
        analysis_run_id=analysis_run_id,
        track_id=track_id,
        event_type=event_type,
        event_state=event_state,
        occurred_from_seconds=occurred_from_seconds,
        occurred_to_seconds=occurred_to_seconds,
    )
    return RuleEventPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_event_response(item) for item in result.items],
    )


def _repository(request: Request) -> RuleRepository:
    return RuleRepository(request.app.state.database.path)


def _rule_response(record: RuleRecord) -> RuleResponse:
    return RuleResponse(**asdict(record))


def _event_response(record: RuleEventRecord) -> RuleEventResponse:
    return RuleEventResponse(**asdict(record))
