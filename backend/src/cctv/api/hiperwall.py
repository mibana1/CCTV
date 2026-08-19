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
from cctv.hiperwall import HiperwallClient, HiperwallRequestError

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


class HiperwallContentInstanceResponse(BaseModel):
    id: str
    position: tuple[float, float] | None = None
    size: tuple[float, float] | None = None
    rotation: float | None = None
    transparency: float | None = None
    rgb: tuple[float, float, float] | None = None
    black_and_white: float | None = None
    mosaic: float | None = None
    layer: float | None = None
    show_label: bool | None = None
    border_rgb: str | None = None
    border_visibility: float | None = None
    audio_volume: float | None = None
    audio_muted: bool | None = None


class HiperwallContentResponse(BaseModel):
    name: str
    type: str
    uuid: str | None = None
    label: str | None = None
    width: float | None = None
    height: float | None = None
    zone_id: str | None = None
    instances: list[HiperwallContentInstanceResponse]


class HiperwallZoneResponse(BaseModel):
    id: str | None = None
    name: str
    left: float | None = None
    top: float | None = None
    width: float | None = None
    height: float | None = None
    color: str | None = None
    grid_horizontal: int | None = None
    grid_vertical: int | None = None


class HiperwallWallResponse(BaseModel):
    name: str
    left: float | None = None
    top: float | None = None
    width: float | None = None
    height: float | None = None
    color: str | None = None
    grid_horizontal: int | None = None
    grid_vertical: int | None = None


class HiperwallInventoryResponse(BaseModel):
    hello: str
    contents: list[HiperwallContentResponse]
    zones: list[HiperwallZoneResponse]
    walls: list[HiperwallWallResponse]


PageNumber = Annotated[int, Query(ge=1)]
PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


@router.get(
    "/hiperwall/inventory",
    response_model=HiperwallInventoryResponse,
    response_model_exclude_none=True,
    responses={
        status.HTTP_502_BAD_GATEWAY: {"description": "Hiperwall inventory request failed"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Hiperwall connection is not configured"
        },
    },
)
def get_hiperwall_inventory(request: Request) -> HiperwallInventoryResponse:
    """Query HiperInterface for selectable content, open instances, walls, and Zones."""
    settings = request.app.state.settings
    if not settings.hiperwall_base_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HIPERWALL_BASE_URL이 설정되지 않았습니다.",
        )
    if settings.hiperwall_auth_mode == "crypto":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="현재 Hiperwall 조회는 none 또는 token 인증만 지원합니다.",
        )
    if settings.hiperwall_auth_mode == "token" and (
        not settings.hiperwall_user
        or settings.hiperwall_token is None
        or not settings.hiperwall_token.get_secret_value()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hiperwall 사용자 또는 토큰이 설정되지 않았습니다.",
        )

    injected_client = getattr(request.app.state, "hiperwall_client", None)
    client = injected_client or HiperwallClient(
        base_url=settings.hiperwall_base_url,
        auth_mode=settings.hiperwall_auth_mode,
        user=settings.hiperwall_user,
        token=(
            settings.hiperwall_token.get_secret_value()
            if settings.hiperwall_token is not None
            else ""
        ),
        timeout_seconds=settings.hiperwall_timeout_seconds,
    )
    try:
        return HiperwallInventoryResponse.model_validate(client.inventory())
    except HiperwallRequestError as error:
        response_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if error.code == "hiperwall_invalid_url"
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(
            status_code=response_status,
            detail=f"Hiperwall 목록 조회 실패: {error}",
        ) from error
    finally:
        if injected_client is None:
            client.close()


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
