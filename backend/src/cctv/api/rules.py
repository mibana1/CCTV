"""Rule configuration and emitted-event API."""

import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from cctv.core.settings import AppMode
from cctv.db import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AnalysisRunStatus,
    DetectionRepository,
    RuleEventRecord,
    RuleRecord,
    RuleRepository,
)
from cctv.hiperwall import (
    HiperwallDryRunPlanner,
    HiperwallLivePlanner,
    HiperwallMappingError,
    validate_rule_hiperwall_mapping,
)
from cctv.hiperwall.mapping import mapping_from_rule
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.rules import RuleConfigurationError, RuleDefinition, RuleEngine, RuleEvent

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


class HiperwallMappingRequest(BaseModel):
    """Rule-specific Hiperwall content, Zone, layout, and point-event duration."""

    enabled: bool = True
    content_name: str | None = Field(default=None, min_length=1, max_length=512)
    content_uuid: str | None = Field(default=None, min_length=1, max_length=512)
    zone_id: str | None = Field(default=None, min_length=1, max_length=512)
    layout: dict[str, Any] | None = None
    display_seconds: int | None = Field(default=None, ge=1, le=86_400)


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


class RuleTestEventResponse(BaseModel):
    rule_event_id: str
    analysis_run_id: str
    display_action_id: str
    mode: str
    status: str
    display_seconds: int


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
        validate_rule_hiperwall_mapping(
            definition,
            default_display_seconds=(
                request.app.state.settings.hiperwall_default_display_seconds
            ),
        )
    except (RuleConfigurationError, HiperwallMappingError) as error:
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


@router.put(
    "/rules/{rule_id}",
    response_model=RuleResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Rule not found"},
        status.HTTP_409_CONFLICT: {"description": "Rule name already exists for source"},
    },
)
def update_rule(
    request: Request,
    rule_id: str,
    body: RuleCreateRequest,
) -> RuleResponse:
    """Validate and replace a dashboard-editable rule configuration."""
    repository = _repository(request)
    if repository.get_rule(rule_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="rule not found")
    definition = RuleDefinition(
        id=rule_id,
        name=body.name,
        source_name=body.source_name,
        rule_type=body.rule_type,
        class_name=body.class_name,
        geometry=body.geometry,
        parameters=body.parameters,
        enabled=body.enabled,
    )
    _validate_definition(request, definition)
    try:
        record = repository.update_rule(
            rule_id,
            name=body.name,
            source_name=body.source_name,
            rule_type=body.rule_type,
            class_name=body.class_name,
            geometry=body.geometry,
            parameters=body.parameters,
            enabled=body.enabled,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="rule not found",
        ) from error
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="a rule with this name already exists for the source",
        ) from error
    return _rule_response(record)


@router.delete(
    "/rules/{rule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Rule not found"}},
)
def delete_rule(request: Request, rule_id: str) -> None:
    """Soft-delete a rule so its historical events and actions remain queryable."""
    try:
        _repository(request).delete_rule(rule_id)
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="rule not found",
        ) from error


@router.post(
    "/rules/{rule_id}/test-event",
    response_model=RuleTestEventResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Rule not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Rule has no enabled Hiperwall mapping"
        },
    },
)
def send_rule_test_event(request: Request, rule_id: str) -> RuleTestEventResponse:
    """Persist an auditable manual event and simulate or queue its mapped action."""
    current = _repository(request).get_rule(rule_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="rule not found")
    definition = current.to_definition()
    settings = request.app.state.settings
    try:
        mapping = mapping_from_rule(
            definition,
            default_display_seconds=settings.hiperwall_default_display_seconds,
        )
    except HiperwallMappingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    if mapping is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Hiperwall 전송 대상이 설정된 이벤트만 테스트할 수 있습니다.",
        )

    event = RuleEvent(
        rule_id=current.id,
        track_id=1,
        event_type="manual_test_triggered",
        event_state="occurred",
        occurred_at_seconds=0,
        class_name=current.class_name or "person",
        confidence=1,
        payload={
            "manual_test": True,
            "rule_type": current.rule_type,
            "trigger": "dashboard",
        },
    )
    repository = DetectionRepository(settings.database_path)
    run = repository.create_analysis_run(
        source_type="rtsp",
        source_name=current.source_name,
        detector_type="manual_rule_test",
        model_name="manual-rule-test",
        model_sha256="0" * 64,
        device="cpu",
        input_size=32,
        confidence_threshold=0.01,
        nms_threshold=0,
        sample_fps=1,
    )
    try:
        planner = (
            HiperwallLivePlanner(settings, (definition,))
            if settings.app_mode is AppMode.LIVE
            else HiperwallDryRunPlanner(settings, (definition,))
        )
        actions = planner.plan(
            (event,),
            analysis_run_id=run.id,
            source_name=current.source_name,
        )
        if len(actions) != 1:
            raise RuntimeError("manual Hiperwall test did not produce one display action")
        repository.save_frame(
            run.id,
            FrameDetections(
                source_index=0,
                sample_index=0,
                timestamp_seconds=0,
                frame_width=1,
                frame_height=1,
                inference_seconds=0,
                detections=(
                    Detection(
                        class_id=0,
                        label=current.class_name or "person",
                        confidence=1,
                        box=BoundingBox(x1=0, y1=0, x2=1, y2=1),
                        track_id=1,
                    ),
                ),
            ),
            active_track_ids={1},
            rule_events=(event,),
            display_actions=actions,
        )
        repository.finish_analysis_run(run.id, status=AnalysisRunStatus.COMPLETED)
    except Exception as error:
        try:
            repository.finish_analysis_run(
                run.id,
                status=AnalysisRunStatus.FAILED,
                error_type=type(error).__name__,
            )
        except LookupError:
            pass
        raise

    worker = getattr(request.app.state, "hiperwall_worker", None)
    if worker is not None:
        worker.wake()
    action = actions[0]
    return RuleTestEventResponse(
        rule_event_id=event.id,
        analysis_run_id=run.id,
        display_action_id=action.id,
        mode=action.mode,
        status=action.status,
        display_seconds=mapping.display_seconds,
    )


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
            validate_rule_hiperwall_mapping(
                current.to_definition(),
                default_display_seconds=(
                    request.app.state.settings.hiperwall_default_display_seconds
                ),
            )
        except (RuleConfigurationError, HiperwallMappingError) as error:
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


@router.put(
    "/rules/{rule_id}/hiperwall",
    response_model=RuleResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Rule not found"}},
)
def set_rule_hiperwall_mapping(
    request: Request,
    rule_id: str,
    body: HiperwallMappingRequest,
) -> RuleResponse:
    """Validate and replace the display mapping used when a rule emits an event."""
    repository = _repository(request)
    current = repository.get_rule(rule_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="rule not found")
    mapping = body.model_dump(exclude_none=True)
    parameters = {**current.parameters, "hiperwall": mapping}
    definition = RuleDefinition(
        id=current.id,
        name=current.name,
        source_name=current.source_name,
        rule_type=current.rule_type,
        class_name=current.class_name,
        geometry=current.geometry,
        parameters=parameters,
        enabled=current.enabled,
    )
    try:
        validate_rule_hiperwall_mapping(
            definition,
            default_display_seconds=(
                request.app.state.settings.hiperwall_default_display_seconds
            ),
        )
    except HiperwallMappingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    return _rule_response(repository.set_rule_hiperwall_mapping(rule_id, mapping))


@router.delete(
    "/rules/{rule_id}/hiperwall",
    response_model=RuleResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Rule not found"}},
)
def clear_rule_hiperwall_mapping(request: Request, rule_id: str) -> RuleResponse:
    """Disable Hiperwall output for a rule without deleting the detection rule."""
    try:
        record = _repository(request).set_rule_hiperwall_mapping(rule_id, None)
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="rule not found",
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


def _validate_definition(request: Request, definition: RuleDefinition) -> None:
    try:
        RuleEngine((definition,))
        validate_rule_hiperwall_mapping(
            definition,
            default_display_seconds=(
                request.app.state.settings.hiperwall_default_display_seconds
            ),
        )
    except (RuleConfigurationError, HiperwallMappingError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


def _rule_response(record: RuleRecord) -> RuleResponse:
    return RuleResponse(**asdict(record))


def _event_response(record: RuleEventRecord) -> RuleEventResponse:
    return RuleEventResponse(**asdict(record))
