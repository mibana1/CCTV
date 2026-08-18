from pathlib import Path

import numpy as np
import pytest

from cctv.db import IdentityRepository, initialize_database
from cctv.identity import (
    SFACE_MODEL_NAME,
    SFACE_MODEL_VERSION,
    DetectedFaceEmbedding,
    FaceBounds,
    FaceIdentityMatcher,
    FaceMatchingConsumer,
    FaceMatchingError,
    FaceMatchRejectionReason,
    FaceMatchStatus,
)
from cctv.media import DecodedFrame


def _repository(tmp_path: Path) -> IdentityRepository:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    return IdentityRepository(database_path)


def _candidates(repository: IdentityRepository):
    return repository.list_embedding_vectors(
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        dimension=3,
    )


def test_matcher_uses_best_photo_score_per_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = repository.create_identity(
        display_name="First person",
        external_id="emp-001",
        identity_id="identity-1",
    )
    second = repository.create_identity(
        display_name="Second person",
        external_id="emp-002",
        identity_id="identity-2",
    )
    repository.add_embedding(
        first.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(1, 0, 0),
        embedding_id="first-front",
    )
    repository.add_embedding(
        first.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(0.8, 0.6, 0),
        embedding_id="first-side",
    )
    repository.add_embedding(
        second.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(0, 1, 0),
        embedding_id="second-front",
    )
    matcher = FaceIdentityMatcher(
        _candidates(repository),
        similarity_threshold=0.8,
        minimum_margin=0.1,
    )

    decision = matcher.match((0.8, 0.6, 0))

    assert decision.matched is True
    assert decision.status is FaceMatchStatus.MATCHED
    assert decision.rejection_reason is None
    assert decision.identity_id == first.id
    assert decision.external_id == "emp-001"
    assert decision.display_name == "First person"
    assert decision.best_candidate.embedding_id == "first-side"
    assert decision.best_candidate.similarity == pytest.approx(1)
    assert decision.second_best_similarity == pytest.approx(0.6)


def test_matcher_rejects_ambiguous_or_low_similarity_faces(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for identity_id, vector in (("identity-a", (1, 0, 0)), ("identity-b", (0, 1, 0))):
        identity = repository.create_identity(display_name=identity_id, identity_id=identity_id)
        repository.add_embedding(
            identity.id,
            model_name=SFACE_MODEL_NAME,
            model_version=SFACE_MODEL_VERSION,
            vector=vector,
        )
    matcher = FaceIdentityMatcher(
        _candidates(repository),
        similarity_threshold=0.7,
        minimum_margin=0.05,
    )

    ambiguous = matcher.match((1, 1, 0))
    low_similarity = matcher.match((0, 0, 1))

    assert ambiguous.best_candidate.similarity == pytest.approx(0.707107)
    assert ambiguous.matched is False
    assert ambiguous.status is FaceMatchStatus.UNKNOWN
    assert ambiguous.rejection_reason is FaceMatchRejectionReason.AMBIGUOUS
    assert ambiguous.identity_id is None
    assert ambiguous.external_id is None
    assert ambiguous.display_name is None
    assert low_similarity.best_candidate.similarity == 0
    assert low_similarity.matched is False
    assert low_similarity.status is FaceMatchStatus.UNKNOWN
    assert low_similarity.rejection_reason is FaceMatchRejectionReason.BELOW_THRESHOLD
    assert low_similarity.identity_id is None


def test_matcher_requires_registered_candidates() -> None:
    with pytest.raises(FaceMatchingError, match="no enabled identity embeddings"):
        FaceIdentityMatcher(())


def test_face_matching_consumer_aggregates_privacy_safe_results(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identity = repository.create_identity(
        display_name="Registered person",
        external_id="emp-001",
        identity_id="identity-1",
    )
    repository.add_embedding(
        identity.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(1, 0, 0),
    )
    matcher = FaceIdentityMatcher(_candidates(repository), similarity_threshold=0.8)

    class FakeExtractor:
        def extract_many(self, image: object, *, source_name: str = "frame"):
            del image, source_name
            return (
                DetectedFaceEmbedding(FaceBounds(1, 2, 30, 40), (1, 0, 0), 0.98),
                DetectedFaceEmbedding(FaceBounds(50, 2, 30, 40), (0, 1, 0), 0.95),
            )

    consumer = FaceMatchingConsumer(FakeExtractor(), matcher, source_name="sample.mp4")
    consumer(
        DecodedFrame(
            source_index=10,
            sample_index=2,
            timestamp_seconds=1.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )
    )

    summary = consumer.summary
    assert summary.processed_frames == 1
    assert summary.frames_with_faces == 1
    assert summary.detected_faces == 2
    assert summary.matched_faces == 1
    assert summary.unknown_faces == 1
    assert summary.matched_identity_counts == {identity.id: 1}
    assert summary.best_similarity == 1
    assert len(consumer.last_observations) == 2
    matched_decision = consumer.last_observations[0].decision
    unknown_decision = consumer.last_observations[1].decision
    assert matched_decision.status is FaceMatchStatus.MATCHED
    assert matched_decision.identity_id == identity.id
    assert unknown_decision.status is FaceMatchStatus.UNKNOWN
    assert unknown_decision.rejection_reason is FaceMatchRejectionReason.BELOW_THRESHOLD
    assert unknown_decision.identity_id is None
    assert unknown_decision.external_id is None
    assert unknown_decision.display_name is None
