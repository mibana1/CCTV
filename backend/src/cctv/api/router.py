"""Top-level API router."""

import logging
import sqlite3
from typing import Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cctv import __version__
from cctv.api.analysis import router as analysis_router
from cctv.api.hiperwall import router as hiperwall_router
from cctv.api.rules import router as rules_router
from cctv.db import check_database_health

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(analysis_router)
router.include_router(hiperwall_router)
router.include_router(rules_router)


class DatabaseHealthResponse(BaseModel):
    """Public SQLite health details that do not expose its filesystem path."""

    status: Literal["ok", "error"]
    schema_version: int | None = None
    journal_mode: str | None = None


class HealthChecksResponse(BaseModel):
    """Critical dependency checks included in the health response."""

    database: DatabaseHealthResponse


class HealthResponse(BaseModel):
    """Application and dependency health response."""

    status: Literal["ok", "error"]
    service: str
    version: str
    environment: str
    mode: str
    checks: HealthChecksResponse


@router.get(
    "/health",
    tags=["system"],
    response_model=HealthResponse,
    response_model_exclude_none=True,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthResponse,
            "description": "A critical dependency is unavailable",
        }
    },
)
def health(request: Request) -> HealthResponse | JSONResponse:
    """Report whether the API and its critical SQLite dependency are ready."""
    settings = request.app.state.settings
    database_state = request.app.state.database

    try:
        database_health = check_database_health(database_state.path)
    except sqlite3.Error as error:
        logger.warning(
            "Health check failed",
            extra={
                "event": "health_check_failed",
                "component": "database",
                "error_type": type(error).__name__,
            },
        )
        response = HealthResponse(
            status="error",
            service="cctv-backend",
            version=__version__,
            environment=settings.app_env,
            mode=settings.app_mode,
            checks=HealthChecksResponse(
                database=DatabaseHealthResponse(status="error"),
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json", exclude_none=True),
        )

    return HealthResponse(
        status="ok",
        service="cctv-backend",
        version=__version__,
        environment=settings.app_env,
        mode=settings.app_mode,
        checks=HealthChecksResponse(
            database=DatabaseHealthResponse(
                status="ok",
                schema_version=database_health.schema_version,
                journal_mode=database_health.journal_mode,
            ),
        ),
    )
