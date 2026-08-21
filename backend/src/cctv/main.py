"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request

from cctv.api.router import router
from cctv.cameras import (
    CameraCredentialCipher,
    CameraManagementService,
    CameraManager,
    CameraReconciler,
    MediaMtxClient,
)
from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import AppMode, Settings, get_settings
from cctv.dashboard import TestSessionManager
from cctv.db import CameraRepository, initialize_database
from cctv.hiperwall import HiperwallActionWorker, HiperwallClient
from cctv.workers import AnalysisSupervisor

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    test_session_manager: TestSessionManager | None = None,
    camera_manager: CameraManager | None = None,
    hiperwall_client: HiperwallClient | None = None,
    analysis_supervisor: AnalysisSupervisor | None = None,
) -> FastAPI:
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
                "app_mode": application_settings.app_mode,
                "ai_device": application_settings.ai_device,
                "analysis_fps": application_settings.analysis_fps,
                "detector_enabled": application_settings.object_detection_enabled,
                "detector_type": application_settings.detector_type,
                "analysis_supervisor_enabled": (application_settings.analysis_supervisor_enabled),
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
        application.state.hiperwall_client = hiperwall_client
        application.state.test_session_manager = test_session_manager or TestSessionManager(
            application_settings
        )
        owned_camera_manager: CameraManagementService | None = None
        camera_reconciler: CameraReconciler | None = None
        hiperwall_worker: HiperwallActionWorker | None = None
        active_analysis_supervisor = analysis_supervisor
        if (
            application_settings.app_mode is AppMode.LIVE
            and application_settings.hiperwall_executor_enabled
        ):
            hiperwall_worker = HiperwallActionWorker(
                application_settings,
                client=hiperwall_client,
            )
            hiperwall_worker.start()
        application.state.hiperwall_worker = hiperwall_worker
        if camera_manager is not None:
            application.state.camera_manager = camera_manager
        elif application_settings.mediamtx_dynamic_paths_enabled:
            cipher = CameraCredentialCipher.from_settings(
                configured_key=application_settings.camera_credential_key,
                key_path=application_settings.camera_credential_key_path,
            )
            owned_camera_manager = CameraManagementService(
                CameraRepository(database_state.path),
                cipher,
                MediaMtxClient(
                    application_settings.mediamtx_api_url,
                    timeout_seconds=application_settings.mediamtx_api_timeout_seconds,
                ),
            )
            application.state.camera_manager = owned_camera_manager
            camera_reconciler = CameraReconciler(
                owned_camera_manager,
                interval_seconds=application_settings.mediamtx_reconcile_interval_seconds,
            )
            camera_reconciler.start()
        else:
            application.state.camera_manager = None
        if active_analysis_supervisor is None and application_settings.analysis_supervisor_enabled:
            active_analysis_supervisor = AnalysisSupervisor(application_settings)
        if active_analysis_supervisor is not None:
            active_analysis_supervisor.start()
        application.state.analysis_supervisor = active_analysis_supervisor
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
            if active_analysis_supervisor is not None:
                active_analysis_supervisor.stop()
            if hiperwall_worker is not None:
                hiperwall_worker.stop()
            if camera_reconciler is not None:
                camera_reconciler.stop()
            if owned_camera_manager is not None:
                owned_camera_manager.close()
            application.state.test_session_manager.shutdown()
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
