"""Operational status API for continuously managed camera analysis workers."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from cctv.workers import AnalysisSupervisor, AnalysisWorkerStatus

router = APIRouter(prefix="/analysis-workers", tags=["analysis-workers"])


class AnalysisWorkerResponse(BaseModel):
    camera_id: int
    camera_name: str
    stream_path: str
    status: AnalysisWorkerStatus
    desired: bool
    lease_owned: bool
    rule_count: int
    configuration_revision: str | None
    analysis_run_id: str | None
    processed_samples: int
    connection_count: int
    reconnect_count: int
    restart_count: int
    started_at: datetime | None
    last_frame_at: datetime | None
    last_event_at: datetime | None
    completed_at: datetime | None
    next_restart_at: datetime | None
    last_error_type: str | None
    last_message: str | None
    requested_device: str | None = None
    effective_device: str | None = None
    execution_provider: str | None = None
    cpu_fallback: bool = False
    fallback_reason: str | None = None


class AnalysisWorkerListResponse(BaseModel):
    enabled: bool
    running: bool
    max_workers: int
    active_workers: int
    last_reconciled_at: datetime | None
    items: list[AnalysisWorkerResponse]


@router.get("", response_model=AnalysisWorkerListResponse)
def list_analysis_workers(request: Request) -> AnalysisWorkerListResponse:
    """Return in-memory ownership, progress, reconnect, and failure state."""
    supervisor = _supervisor(request)
    if supervisor is None:
        settings = request.app.state.settings
        return AnalysisWorkerListResponse(
            enabled=settings.analysis_supervisor_enabled,
            running=False,
            max_workers=settings.analysis_supervisor_max_workers,
            active_workers=0,
            last_reconciled_at=None,
            items=[],
        )
    snapshot = supervisor.snapshot()
    return AnalysisWorkerListResponse(
        enabled=snapshot.enabled,
        running=snapshot.running,
        max_workers=snapshot.max_workers,
        active_workers=snapshot.active_workers,
        last_reconciled_at=snapshot.last_reconciled_at,
        items=[
            AnalysisWorkerResponse.model_validate(item, from_attributes=True)
            for item in snapshot.items
        ],
    )


@router.get(
    "/{camera_id}",
    response_model=AnalysisWorkerResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Analysis worker not found"}},
)
def get_analysis_worker(request: Request, camera_id: int) -> AnalysisWorkerResponse:
    """Return one camera's current managed-analysis state."""
    supervisor = _supervisor(request)
    if supervisor is not None:
        for item in supervisor.snapshot().items:
            if item.camera_id == camera_id:
                return AnalysisWorkerResponse.model_validate(item, from_attributes=True)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="analysis worker not found",
    )


def _supervisor(request: Request) -> AnalysisSupervisor | None:
    return getattr(request.app.state, "analysis_supervisor", None)


__all__ = ["router"]
