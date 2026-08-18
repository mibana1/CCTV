"""FastAPI application entry point."""

import uvicorn
from fastapi import FastAPI

from cctv.api.router import router

app = FastAPI(
    title="CCTV Analysis API",
    version="0.1.0",
)
app.include_router(router)


def run() -> None:
    """Run the local development API."""
    uvicorn.run("cctv.main:app", host="127.0.0.1", port=8000)

