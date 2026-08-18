"""Top-level API router."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report whether the API process is available."""
    return {"status": "ok"}

