"""Atomic identity registration from a small local photo set."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cctv.db import (
    IdentityEmbeddingInput,
    IdentityEmbeddingRecord,
    IdentityRecord,
    IdentityRepository,
)
from cctv.identity.face import ExtractedFaceEmbedding, FaceEmbeddingModelMetadata

logger = logging.getLogger(__name__)

MIN_REGISTRATION_PHOTOS = 3
MAX_REGISTRATION_PHOTOS = 5
SUPPORTED_PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


class FaceRegistrationError(RuntimeError):
    """A photo set or target identity cannot be registered safely."""


class FaceEmbeddingExtractor(Protocol):
    metadata: FaceEmbeddingModelMetadata

    def extract_file(self, photo_path: str | Path) -> ExtractedFaceEmbedding: ...


@dataclass(frozen=True, slots=True)
class RegisteredPhotoEmbedding:
    photo_name: str
    detection_confidence: float
    embedding: IdentityEmbeddingRecord


@dataclass(frozen=True, slots=True)
class FaceRegistrationResult:
    identity: IdentityRecord
    photo_directory: Path
    items: tuple[RegisteredPhotoEmbedding, ...]


class FaceRegistrationService:
    """Generate all photo embeddings before atomically committing the batch."""

    def __init__(
        self,
        repository: IdentityRepository,
        extractor: FaceEmbeddingExtractor,
        *,
        photo_root: str | Path,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.photo_root = Path(photo_root).expanduser().resolve()

    def register(
        self,
        identity_id: str,
        *,
        photo_directory: str | Path | None = None,
    ) -> FaceRegistrationResult:
        identity = self.repository.get_identity(identity_id)
        if identity is None:
            raise FaceRegistrationError(f"identity does not exist: {identity_id}")
        directory = (
            Path(photo_directory).expanduser().resolve()
            if photo_directory is not None
            else self.photo_root / _default_photo_set_name(identity)
        )
        photos = discover_registration_photos(directory)

        logger.info(
            "Face registration started",
            extra={
                "event": "face_registration_started",
                "identity_id": identity.id,
                "photo_count": len(photos),
                "embedding_model": self.extractor.metadata.model_name,
                "embedding_model_version": self.extractor.metadata.model_version,
            },
        )
        extracted = tuple(self.extractor.extract_file(photo) for photo in photos)
        dimensions = {len(item.vector) for item in extracted}
        if dimensions != {self.extractor.metadata.dimension}:
            raise FaceRegistrationError(
                "registration photos produced inconsistent embedding dimensions"
            )

        records = self.repository.add_embeddings(
            identity.id,
            tuple(
                IdentityEmbeddingInput(
                    model_name=self.extractor.metadata.model_name,
                    model_version=self.extractor.metadata.model_version,
                    vector=item.vector,
                    source_reference=photo.name,
                    quality_score=item.detection_confidence,
                )
                for photo, item in zip(photos, extracted, strict=True)
            ),
        )
        result = FaceRegistrationResult(
            identity=identity,
            photo_directory=directory,
            items=tuple(
                RegisteredPhotoEmbedding(
                    photo_name=photo.name,
                    detection_confidence=item.detection_confidence,
                    embedding=record,
                )
                for photo, item, record in zip(photos, extracted, records, strict=True)
            ),
        )
        logger.info(
            "Face registration completed",
            extra={
                "event": "face_registration_completed",
                "identity_id": identity.id,
                "photo_count": len(result.items),
                "embedding_model": self.extractor.metadata.model_name,
                "embedding_model_version": self.extractor.metadata.model_version,
                "embedding_dimension": self.extractor.metadata.dimension,
            },
        )
        return result


def discover_registration_photos(directory: str | Path) -> tuple[Path, ...]:
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise FaceRegistrationError(f"registration photo directory does not exist: {path}")
    photos = tuple(
        sorted(
            (
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.casefold() in SUPPORTED_PHOTO_SUFFIXES
            ),
            key=lambda item: item.name.casefold(),
        )
    )
    if not MIN_REGISTRATION_PHOTOS <= len(photos) <= MAX_REGISTRATION_PHOTOS:
        raise FaceRegistrationError(
            f"registration requires {MIN_REGISTRATION_PHOTOS} to "
            f"{MAX_REGISTRATION_PHOTOS} JPG or PNG photos; found {len(photos)}"
        )
    return photos


def _default_photo_set_name(identity: IdentityRecord) -> str:
    value = identity.external_id or identity.id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise FaceRegistrationError(
            "external_id is not safe for an automatic photo directory; use --photo-dir"
        )
    return value


__all__ = [
    "MAX_REGISTRATION_PHOTOS",
    "MIN_REGISTRATION_PHOTOS",
    "SUPPORTED_PHOTO_SUFFIXES",
    "FaceEmbeddingExtractor",
    "FaceRegistrationError",
    "FaceRegistrationResult",
    "FaceRegistrationService",
    "RegisteredPhotoEmbedding",
    "discover_registration_photos",
]
