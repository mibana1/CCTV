"""Runtime construction for the default YuNet/SFace face matching pipeline."""

from __future__ import annotations

import logging

from cctv.core.settings import Settings
from cctv.db import IdentityRepository, initialize_database
from cctv.identity.face import OpenCvSFaceExtractor
from cctv.identity.matching import FaceIdentityMatcher, FaceMatchingConsumer

logger = logging.getLogger(__name__)


def create_sface_matching_consumer(
    settings: Settings,
    *,
    source_name: str,
    similarity_threshold: float | None = None,
    minimum_margin: float | None = None,
) -> FaceMatchingConsumer:
    """Load enabled registered identities and build one sequential frame consumer."""
    initialize_database(settings.database_path)
    extractor = OpenCvSFaceExtractor(
        settings.face_detection_model_path,
        settings.face_embedding_model_path,
        score_threshold=settings.face_detection_score_threshold,
        nms_threshold=settings.face_detection_nms_threshold,
        top_k=settings.face_detection_top_k,
        max_input_dimension=settings.face_detection_max_input_dimension,
    )
    metadata = extractor.metadata
    candidates = IdentityRepository(settings.database_path).list_embedding_vectors(
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        dimension=metadata.dimension,
        enabled_identities_only=True,
    )
    matcher = FaceIdentityMatcher(
        candidates,
        similarity_threshold=(
            similarity_threshold
            if similarity_threshold is not None
            else settings.face_match_similarity_threshold
        ),
        minimum_margin=(
            minimum_margin if minimum_margin is not None else settings.face_match_minimum_margin
        ),
    )
    logger.info(
        "Face matching pipeline initialized",
        extra={
            "event": "face_matching_pipeline_initialized",
            "source_name": source_name,
            "model_name": metadata.model_name,
            "model_version": metadata.model_version,
            "embedding_dimension": metadata.dimension,
            "candidate_identity_count": matcher.identity_count,
            "candidate_embedding_count": len(matcher.candidates),
            "similarity_threshold": matcher.similarity_threshold,
            "minimum_margin": matcher.minimum_margin,
        },
    )
    return FaceMatchingConsumer(extractor, matcher, source_name=source_name)


__all__ = ["create_sface_matching_consumer"]
