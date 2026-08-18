"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request

from cctv.api.router import router
from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import Settings, get_settings
from cctv.db import initialize_database

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application with validated settings and database startup."""
    application_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            level=application_settings.log_level,
            log_path=application_settings.log_path,
            max_bytes=application_settings.log_max_bytes,
            backup_count=application_settings.log_backup_count,
        )
        logger.info(
            "Application startup started",
            extra={
                "event": "application_starting",
                "app_env": application_settings.app_env,
            },
        )
        try:
            database_state = initialize_database(application_settings.database_path)
        except Exception:
            logger.exception(
                "Database initialization failed",
                extra={
                    "event": "database_initialization_failed",
                    "database_path": application_settings.database_path,
                },
            )
            shutdown_logging()
            raise

        application.state.settings = application_settings
        application.state.database = database_state
        logger.info(
            "Application startup complete",
            extra={
                "event": "application_started",
                "schema_version": database_state.schema_version,
            },
        )
        try:
            yield
        finally:
            logger.info(
                "Application shutdown complete",
                extra={"event": "application_stopped"},
            )
            shutdown_logging()

    application = FastAPI(
        title="CCTV Analysis API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def log_http_request(request: Request, call_next):
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "HTTP request failed",
                extra={
                    "event": "http_request_failed",
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": round((perf_counter() - started_at) * 1_000, 3),
                },
            )
            raise

        logger.info(
            "HTTP request completed",
            extra={
                "event": "http_request_completed",
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started_at) * 1_000, 3),
            },
        )
        return response

    application.include_router(router)
    return application


app = create_app()


def run() -> None:
    """Run the local development API."""
    settings = get_settings()
    uvicorn.run(
        "cctv.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
