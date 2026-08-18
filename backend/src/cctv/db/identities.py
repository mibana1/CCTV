"""SQLite repository for registered people and model-neutral embeddings."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from collections.abc import Collection, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4

from cctv.db.database import connect_database
from cctv.db.detections import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

MIN_EMBEDDING_DIMENSIONS = 2
MAX_EMBEDDING_DIMENSIONS = 4096
DEFAULT_MODEL_VERSION = "unspecified"


@dataclass(frozen=True, slots=True)
class IdentityRecord:
    """One registered person without biometric vector data."""

    id: str
    external_id: str | None
    display_name: str
    description: str | None
    metadata: dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class IdentityEmbeddingRecord:
    """Safe embedding metadata returned by management APIs."""

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


@dataclass(frozen=True, slots=True)
class IdentityEmbeddingVector:
    """Internal matching input including the normalized vector."""

    identity: IdentityRecord
    embedding: IdentityEmbeddingRecord
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class IdentityPage:
    items: tuple[IdentityRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class IdentityEmbeddingPage:
    items: tuple[IdentityEmbeddingRecord, ...]
    total: int


class IdentityRepository:
    """Register people and manage embeddings independently of an inference model."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def create_identity(
        self,
        *,
        display_name: str,
        external_id: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        enabled: bool = True,
        identity_id: str | None = None,
    ) -> IdentityRecord:
        identifier = identity_id or str(uuid4())
        with closing(connect_database(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO identities (
                    id, external_id, display_name, description, metadata_json, enabled
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    _optional_text(external_id, "external_id"),
                    _required_text(display_name, "display_name"),
                    _optional_text(description, "description"),
                    _json_object(metadata or {}, "metadata"),
                    int(enabled),
                ),
            )
            row = connection.execute(
                "SELECT * FROM identities WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise RuntimeError("identity was not created")
        return _identity_from_row(row)

    def get_identity(self, identity_id: str) -> IdentityRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM identities WHERE id = ?",
                (identity_id,),
            ).fetchone()
        return _identity_from_row(row) if row is not None else None

    def list_identities(
        self,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        query: str | None = None,
        enabled: bool | None = None,
    ) -> IdentityPage:
        _validate_pagination(page, limit)
        clauses: list[str] = []
        parameters: list[object] = []
        if query is not None:
            normalized_query = _required_text(query, "query")
            clauses.append(
                "(display_name LIKE ? COLLATE NOCASE OR external_id LIKE ? COLLATE NOCASE)"
            )
            pattern = f"%{normalized_query}%"
            parameters.extend((pattern, pattern))
        if enabled is not None:
            clauses.append("enabled = ?")
            parameters.append(int(enabled))
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM identities{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM identities{where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return IdentityPage(
            items=tuple(_identity_from_row(row) for row in rows),
            total=total,
        )

    def update_identity(
        self,
        identity_id: str,
        changes: Mapping[str, Any],
    ) -> IdentityRecord:
        allowed = {"display_name", "external_id", "description", "metadata", "enabled"}
        unexpected = set(changes) - allowed
        if unexpected:
            raise ValueError(f"unsupported identity fields: {', '.join(sorted(unexpected))}")
        if not changes:
            raise ValueError("at least one identity field is required")

        assignments: list[str] = []
        parameters: list[object] = []
        for field, value in changes.items():
            if field == "display_name":
                assignments.append("display_name = ?")
                parameters.append(_required_text(value, field))
            elif field in {"external_id", "description"}:
                assignments.append(f"{field} = ?")
                parameters.append(_optional_text(value, field))
            elif field == "metadata":
                assignments.append("metadata_json = ?")
                parameters.append(_json_object(value, field))
            elif field == "enabled":
                if not isinstance(value, bool):
                    raise TypeError("enabled must be a boolean")
                assignments.append("enabled = ?")
                parameters.append(int(value))
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        parameters.append(identity_id)

        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                f"UPDATE identities SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise LookupError(f"identity does not exist: {identity_id}")
            row = connection.execute(
                "SELECT * FROM identities WHERE id = ?",
                (identity_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("updated identity could not be loaded")
        return _identity_from_row(row)

    def delete_identity(self, identity_id: str) -> bool:
        """Delete a registration and cascade its embeddings."""
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
        return cursor.rowcount > 0

    def add_embedding(
        self,
        identity_id: str,
        *,
        model_name: str,
        vector: Sequence[float],
        model_version: str = DEFAULT_MODEL_VERSION,
        source_reference: str | None = None,
        quality_score: float | None = None,
        embedding_id: str | None = None,
    ) -> IdentityEmbeddingRecord:
        normalized_vector = normalize_embedding(vector)
        vector_blob = _pack_vector(normalized_vector)
        identifier = embedding_id or str(uuid4())
        normalized_quality = _quality_score(quality_score)

        with closing(connect_database(self.database_path)) as connection, connection:
            identity = connection.execute(
                "SELECT id FROM identities WHERE id = ?",
                (identity_id,),
            ).fetchone()
            if identity is None:
                raise LookupError(f"identity does not exist: {identity_id}")
            connection.execute(
                """
                INSERT INTO identity_embeddings (
                    id, identity_id, model_name, model_version, dimension,
                    embedding_sha256, vector_blob, source_reference, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    identity_id,
                    _required_text(model_name, "model_name"),
                    _required_text(model_version, "model_version"),
                    len(normalized_vector),
                    hashlib.sha256(vector_blob).hexdigest(),
                    vector_blob,
                    _optional_text(source_reference, "source_reference"),
                    normalized_quality,
                ),
            )
            row = connection.execute(
                "SELECT * FROM identity_embeddings WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise RuntimeError("identity embedding was not created")
        return _embedding_from_row(row)

    def get_embedding(self, embedding_id: str) -> IdentityEmbeddingRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM identity_embeddings WHERE id = ?",
                (embedding_id,),
            ).fetchone()
        return _embedding_from_row(row) if row is not None else None

    def list_embeddings(
        self,
        identity_id: str,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> IdentityEmbeddingPage:
        _validate_pagination(page, limit)
        clauses = ["identity_id = ?"]
        parameters: list[object] = [identity_id]
        for column, value in (
            ("model_name", model_name),
            ("model_version", model_version),
        ):
            if value is not None:
                clauses.append(f"{column} = ? COLLATE NOCASE")
                parameters.append(_required_text(value, column))
        where_sql = f" WHERE {' AND '.join(clauses)}"
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM identities WHERE id = ?",
                    (identity_id,),
                ).fetchone()
                is None
            ):
                raise LookupError(f"identity does not exist: {identity_id}")
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM identity_embeddings{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM identity_embeddings{where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return IdentityEmbeddingPage(
            items=tuple(_embedding_from_row(row) for row in rows),
            total=total,
        )

    def list_embedding_vectors(
        self,
        *,
        model_name: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        dimension: int | None = None,
        enabled_identities_only: bool = True,
    ) -> tuple[IdentityEmbeddingVector, ...]:
        """Load internal candidate vectors for a future matching service."""
        normalized_model_name = _required_text(model_name, "model_name")
        normalized_model_version = _required_text(model_version, "model_version")
        clauses = [
            "embedding.model_name = ? COLLATE NOCASE",
            "embedding.model_version = ? COLLATE NOCASE",
        ]
        parameters: list[object] = [normalized_model_name, normalized_model_version]
        if dimension is not None:
            if not MIN_EMBEDDING_DIMENSIONS <= dimension <= MAX_EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"dimension must be between {MIN_EMBEDDING_DIMENSIONS} "
                    f"and {MAX_EMBEDDING_DIMENSIONS}"
                )
            clauses.append("embedding.dimension = ?")
            parameters.append(dimension)
        if enabled_identities_only:
            clauses.append("identity.enabled = 1")
        with closing(connect_database(self.database_path)) as connection:
            rows = connection.execute(
                f"""
                SELECT
                    identity.*,
                    embedding.id AS embedding_id,
                    embedding.identity_id AS embedding_identity_id,
                    embedding.model_name,
                    embedding.model_version,
                    embedding.dimension,
                    embedding.dtype,
                    embedding.normalized,
                    embedding.embedding_sha256,
                    embedding.vector_blob,
                    embedding.source_reference,
                    embedding.quality_score,
                    embedding.created_at AS embedding_created_at
                FROM identity_embeddings AS embedding
                JOIN identities AS identity ON identity.id = embedding.identity_id
                WHERE {" AND ".join(clauses)}
                ORDER BY identity.id, embedding.id
                """,
                parameters,
            ).fetchall()
        return tuple(
            IdentityEmbeddingVector(
                identity=_identity_from_row(row),
                embedding=_embedding_from_joined_row(row),
                vector=_unpack_vector(row["vector_blob"], int(row["dimension"])),
            )
            for row in rows
        )

    def delete_embedding(self, identity_id: str, embedding_id: str) -> bool:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM identity_embeddings WHERE id = ? AND identity_id = ?",
                (embedding_id, identity_id),
            )
        return cursor.rowcount > 0


def normalize_embedding(values: Collection[float]) -> tuple[float, ...]:
    """Validate and L2-normalize a finite embedding vector."""
    vector = tuple(float(value) for value in values)
    if not MIN_EMBEDDING_DIMENSIONS <= len(vector) <= MAX_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"embedding dimension must be between {MIN_EMBEDDING_DIMENSIONS} "
            f"and {MAX_EMBEDDING_DIMENSIONS}"
        )
    if any(not isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    norm = sqrt(sum(value * value for value in vector))
    if not isfinite(norm) or norm <= 0:
        raise ValueError("embedding vector must have a non-zero finite L2 norm")
    return tuple(value / norm for value in vector)


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(value: bytes, dimension: int) -> tuple[float, ...]:
    return tuple(struct.unpack(f"<{dimension}f", value))


def _quality_score(value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError("quality_score must be a finite number between 0 and 1")
    return normalized


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _optional_text(value: object | None, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _json_object(value: Mapping[str, Any], name: str) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JSON-compatible values") from error


def _json_from_text(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("stored identity metadata must be an object")
    return parsed


def _datetime_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _identity_from_row(row: sqlite3.Row) -> IdentityRecord:
    return IdentityRecord(
        id=str(row["id"]),
        external_id=str(row["external_id"]) if row["external_id"] is not None else None,
        display_name=str(row["display_name"]),
        description=str(row["description"]) if row["description"] is not None else None,
        metadata=_json_from_text(str(row["metadata_json"])),
        enabled=bool(row["enabled"]),
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _embedding_from_row(row: sqlite3.Row) -> IdentityEmbeddingRecord:
    return IdentityEmbeddingRecord(
        id=str(row["id"]),
        identity_id=str(row["identity_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        dimension=int(row["dimension"]),
        dtype=str(row["dtype"]),
        normalized=bool(row["normalized"]),
        embedding_sha256=str(row["embedding_sha256"]),
        source_reference=(
            str(row["source_reference"]) if row["source_reference"] is not None else None
        ),
        quality_score=(float(row["quality_score"]) if row["quality_score"] is not None else None),
        created_at=_datetime_from_text(str(row["created_at"])),
    )


def _embedding_from_joined_row(row: sqlite3.Row) -> IdentityEmbeddingRecord:
    return IdentityEmbeddingRecord(
        id=str(row["embedding_id"]),
        identity_id=str(row["embedding_identity_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        dimension=int(row["dimension"]),
        dtype=str(row["dtype"]),
        normalized=bool(row["normalized"]),
        embedding_sha256=str(row["embedding_sha256"]),
        source_reference=(
            str(row["source_reference"]) if row["source_reference"] is not None else None
        ),
        quality_score=(float(row["quality_score"]) if row["quality_score"] is not None else None),
        created_at=_datetime_from_text(str(row["embedding_created_at"])),
    )


def _validate_pagination(page: int, limit: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


__all__ = [
    "DEFAULT_MODEL_VERSION",
    "MAX_EMBEDDING_DIMENSIONS",
    "MIN_EMBEDDING_DIMENSIONS",
    "IdentityEmbeddingPage",
    "IdentityEmbeddingRecord",
    "IdentityEmbeddingVector",
    "IdentityPage",
    "IdentityRecord",
    "IdentityRepository",
    "normalize_embedding",
]
