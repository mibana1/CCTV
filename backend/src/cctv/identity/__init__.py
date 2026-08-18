"""Registered-person face embedding extraction and enrollment."""

from cctv.identity.face import (
    SFACE_EMBEDDING_DIMENSION,
    SFACE_MODEL_NAME,
    SFACE_MODEL_VERSION,
    YUNET_MODEL_NAME,
    YUNET_MODEL_VERSION,
    ExtractedFaceEmbedding,
    FaceEmbeddingError,
    FaceEmbeddingModelMetadata,
    FaceModelLoadError,
    FacePhotoError,
    OpenCvSFaceExtractor,
)
from cctv.identity.registration import (
    MAX_REGISTRATION_PHOTOS,
    MIN_REGISTRATION_PHOTOS,
    SUPPORTED_PHOTO_SUFFIXES,
    FaceEmbeddingExtractor,
    FaceRegistrationError,
    FaceRegistrationResult,
    FaceRegistrationService,
    RegisteredPhotoEmbedding,
    discover_registration_photos,
)

__all__ = [
    "MAX_REGISTRATION_PHOTOS",
    "MIN_REGISTRATION_PHOTOS",
    "SFACE_EMBEDDING_DIMENSION",
    "SFACE_MODEL_NAME",
    "SFACE_MODEL_VERSION",
    "SUPPORTED_PHOTO_SUFFIXES",
    "YUNET_MODEL_NAME",
    "YUNET_MODEL_VERSION",
    "ExtractedFaceEmbedding",
    "FaceEmbeddingError",
    "FaceEmbeddingExtractor",
    "FaceEmbeddingModelMetadata",
    "FaceModelLoadError",
    "FacePhotoError",
    "FaceRegistrationError",
    "FaceRegistrationResult",
    "FaceRegistrationService",
    "OpenCvSFaceExtractor",
    "RegisteredPhotoEmbedding",
    "discover_registration_photos",
]
