"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from cctv.api.router import router
from cctv.core.settings import Settings, get_settings
from cctv.db import initialize_database


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an application with validated settings and database startup."""
    application_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database_state = initialize_database(application_settings.database_path)
        application.state.settings = application_settings
        application.state.database = database_state
        yield

    application = FastAPI(
        title="CCTV Analysis API",
        version="0.1.0",
        lifespan=lifespan,
    )
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
    )
