"""Management API for registered people and model-neutral embeddings."""

import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field, model_validator

from cctv.db import (
    DEFAULT_PAGE_SIZE,
    MAX_EMBEDDING_DIMENSIONS,
    MAX_PAGE_SIZE,
    MIN_EMBEDDING_DIMENSIONS,
    IdentityEmbeddingRecord,
    IdentityRecord,
    IdentityRepository,
)

router = APIRouter(tags=["identities"])


class IdentityCreateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)
    external_id: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, min_length=1, max_length=1_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class IdentityUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    external_id: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, min_length=1, max_length=1_000)
    metadata: dict[str, Any] | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> "IdentityUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("at least one identity field is required")
        return self


class IdentityResponse(BaseModel):
    id: str
    external_id: str | None
    display_name: str
    description: str | None
    metadata: dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime


class IdentityPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[IdentityResponse]


class IdentityEmbeddingCreateRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(default="unspecified", min_length=1, max_length=128)
    vector: list[float] = Field(
        min_length=MIN_EMBEDDING_DIMENSIONS,
        max_length=MAX_EMBEDDING_DIMENSIONS,
    )
    source_reference: str | None = Field(default=None, min_length=1, max_length=1_000)
    quality_score: float | None = Field(default=None, ge=0, le=1)


class IdentityEmbeddingResponse(BaseModel):
    """Embedding metadata; the biometric vector is intentionally omitted."""

    id: str
    identity_id: str
    model_name: str
    model_version: str
    dimension: int
    dtype: str
    normalized: bool
    embedding_sha256: str
    source_reference: str | None
    quality_score: float | None
    created_at: datetime


class IdentityEmbeddingPageResponse(BaseModel):
    page: int
    limit: int
    total: int
    has_next: bool
    items: list[IdentityEmbeddingResponse]


PageNumber = Annotated[int, Query(ge=1)]
PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)]


@router.post(
    "/identities",
    response_model=IdentityResponse,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "External ID already registered"}},
)
def create_identity(request: Request, body: IdentityCreateRequest) -> IdentityResponse:
    try:
        record = _repository(request).create_identity(
            display_name=body.display_name,
            external_id=body.external_id,
            description=body.description,
            metadata=body.metadata,
            enabled=body.enabled,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="external_id is already registered",
        ) from error
    return _identity_response(record)


@router.get("/identities", response_model=IdentityPageResponse)
def list_identities(
    request: Request,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    query: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    enabled: Annotated[bool | None, Query()] = None,
) -> IdentityPageResponse:
    try:
        result = _repository(request).list_identities(
            page=page,
            limit=limit,
            query=query,
            enabled=enabled,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    return IdentityPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_identity_response(item) for item in result.items],
    )


@router.get(
    "/identities/{identity_id}",
    response_model=IdentityResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Identity not found"}},
)
def get_identity(request: Request, identity_id: str) -> IdentityResponse:
    record = _repository(request).get_identity(identity_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="identity not found")
    return _identity_response(record)


@router.patch(
    "/identities/{identity_id}",
    response_model=IdentityResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Identity not found"},
        status.HTTP_409_CONFLICT: {"description": "External ID already registered"},
    },
)
def update_identity(
    request: Request,
    identity_id: str,
    body: IdentityUpdateRequest,
) -> IdentityResponse:
    try:
        record = _repository(request).update_identity(
            identity_id,
            body.model_dump(exclude_unset=True),
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="identity not found",
        ) from error
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="external_id is already registered",
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    return _identity_response(record)


@router.delete(
    "/identities/{identity_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Identity not found"}},
)
def delete_identity(request: Request, identity_id: str) -> Response:
    if not _repository(request).delete_identity(identity_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="identity not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/identities/{identity_id}/embeddings",
    response_model=IdentityEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Identity not found"},
        status.HTTP_409_CONFLICT: {"description": "Embedding already registered"},
    },
)
def create_identity_embedding(
    request: Request,
    identity_id: str,
    body: IdentityEmbeddingCreateRequest,
) -> IdentityEmbeddingResponse:
    try:
        record = _repository(request).add_embedding(
            identity_id,
            model_name=body.model_name,
            model_version=body.model_version,
            vector=body.vector,
            source_reference=body.source_reference,
            quality_score=body.quality_score,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="identity not found",
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except sqlite3.IntegrityError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="the same embedding is already registered for this identity and model",
        ) from error
    return _embedding_response(record)


@router.get(
    "/identities/{identity_id}/embeddings",
    response_model=IdentityEmbeddingPageResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Identity not found"}},
)
def list_identity_embeddings(
    request: Request,
    identity_id: str,
    page: PageNumber = 1,
    limit: PageLimit = DEFAULT_PAGE_SIZE,
    model_name: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    model_version: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> IdentityEmbeddingPageResponse:
    try:
        result = _repository(request).list_embeddings(
            identity_id,
            page=page,
            limit=limit,
            model_name=model_name,
            model_version=model_version,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="identity not found",
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    return IdentityEmbeddingPageResponse(
        page=page,
        limit=limit,
        total=result.total,
        has_next=page * limit < result.total,
        items=[_embedding_response(item) for item in result.items],
    )


@router.delete(
    "/identities/{identity_id}/embeddings/{embedding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Embedding not found"}},
)
def delete_identity_embedding(
    request: Request,
    identity_id: str,
    embedding_id: str,
) -> Response:
    if not _repository(request).delete_embedding(identity_id, embedding_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="embedding not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _repository(request: Request) -> IdentityRepository:
    return IdentityRepository(request.app.state.database.path)


def _identity_response(record: IdentityRecord) -> IdentityResponse:
    return IdentityResponse(**asdict(record))


def _embedding_response(record: IdentityEmbeddingRecord) -> IdentityEmbeddingResponse:
    return IdentityEmbeddingResponse(**asdict(record))
