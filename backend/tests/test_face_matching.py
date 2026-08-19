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
from cctv.inference import BoundingBox, Detection, FrameDetections
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


def test_face_matching_consumer_reuses_matched_result_for_same_track(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identity = repository.create_identity(display_name="Person", identity_id="person-1")
    repository.add_embedding(
        identity.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(1, 0, 0),
    )

    class CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0
            self.shapes: list[tuple[int, ...]] = []

        def extract_many(self, image: np.ndarray, *, source_name: str = "frame"):
            del source_name
            self.calls += 1
            self.shapes.append(image.shape)
            return (DetectedFaceEmbedding(FaceBounds(1, 2, 20, 20), (1, 0, 0), 0.95),)

    extractor = CountingExtractor()
    consumer = FaceMatchingConsumer(
        extractor,
        FaceIdentityMatcher(_candidates(repository), similarity_threshold=0.8),
        source_name="camera",
        unknown_retry_seconds=2,
    )

    for sample_index in range(3):
        frame, detections = _tracked_person_frame(sample_index, float(sample_index), track_id=7)
        consumer.process_tracked(frame, detections, active_track_ids=(7,))
        if sample_index == 0:
            assert consumer.last_observations[0].face.bounds == FaceBounds(11, 22, 20, 20)

    summary = consumer.summary
    assert extractor.calls == 1
    assert extractor.shapes == [(80, 80, 3)]
    assert summary.track_cache_enabled is True
    assert summary.processed_frames == 3
    assert summary.tracked_person_detections == 3
    assert summary.face_analysis_attempts == 1
    assert summary.cache_hits == 2
    assert summary.cached_tracks == 1
    assert summary.matched_faces == 1
    assert summary.matched_identity_counts == {identity.id: 1}
    assert consumer.get_cached_decision(7) is not None
    assert consumer.get_cached_decision(7).identity_id == identity.id
    assert consumer.get_cached_decision(999) is None

    expired_frame = DecodedFrame(
        source_index=3,
        sample_index=3,
        timestamp_seconds=3,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    consumer.process_tracked(
        expired_frame,
        FrameDetections(
            source_index=3,
            sample_index=3,
            timestamp_seconds=3,
            frame_width=100,
            frame_height=100,
            inference_seconds=0,
            detections=(),
        ),
        active_track_ids=(),
    )
    assert consumer.get_cached_decision(7) is None
    assert consumer.summary.cached_tracks == 0


def test_unknown_track_is_reanalyzed_after_retry_interval(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identity = repository.create_identity(display_name="Person", identity_id="person-1")
    repository.add_embedding(
        identity.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(1, 0, 0),
    )

    class ImprovingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract_many(self, image: np.ndarray, *, source_name: str = "frame"):
            del image, source_name
            self.calls += 1
            vector = (0, 1, 0) if self.calls == 1 else (1, 0, 0)
            return (DetectedFaceEmbedding(FaceBounds(1, 2, 20, 20), vector, 0.95),)

    extractor = ImprovingExtractor()
    consumer = FaceMatchingConsumer(
        extractor,
        FaceIdentityMatcher(_candidates(repository), similarity_threshold=0.8),
        source_name="camera",
        unknown_retry_seconds=2,
    )

    observed_statuses: list[FaceMatchStatus] = []
    for sample_index in range(4):
        frame, detections = _tracked_person_frame(sample_index, float(sample_index), track_id=7)
        consumer.process_tracked(frame, detections, active_track_ids=(7,))
        if consumer.last_observations:
            observed_statuses.append(consumer.last_observations[0].decision.status)

    summary = consumer.summary
    assert extractor.calls == 2
    assert observed_statuses == [FaceMatchStatus.UNKNOWN, FaceMatchStatus.MATCHED]
    assert summary.face_analysis_attempts == 2
    assert summary.cache_hits == 2
    assert summary.unknown_faces == 1
    assert summary.matched_faces == 1


def test_face_not_detected_is_temporarily_cached(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identity = repository.create_identity(display_name="Person", identity_id="person-1")
    repository.add_embedding(
        identity.id,
        model_name=SFACE_MODEL_NAME,
        model_version=SFACE_MODEL_VERSION,
        vector=(1, 0, 0),
    )

    class EmptyExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract_many(self, image: np.ndarray, *, source_name: str = "frame"):
            del image, source_name
            self.calls += 1
            return ()

    extractor = EmptyExtractor()
    consumer = FaceMatchingConsumer(
        extractor,
        FaceIdentityMatcher(_candidates(repository)),
        source_name="camera",
        unknown_retry_seconds=2,
    )

    for sample_index in range(3):
        frame, detections = _tracked_person_frame(sample_index, float(sample_index), track_id=7)
        consumer.process_tracked(frame, detections, active_track_ids=(7,))

    assert extractor.calls == 2
    assert consumer.summary.face_analysis_attempts == 2
    assert consumer.summary.cache_hits == 1
    assert consumer.summary.detected_faces == 0


def _tracked_person_frame(
    sample_index: int,
    timestamp_seconds: float,
    *,
    track_id: int,
) -> tuple[DecodedFrame, FrameDetections]:
    frame = DecodedFrame(
        source_index=sample_index,
        sample_index=sample_index,
        timestamp_seconds=timestamp_seconds,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    result = FrameDetections(
        source_index=sample_index,
        sample_index=sample_index,
        timestamp_seconds=timestamp_seconds,
        frame_width=100,
        frame_height=100,
        inference_seconds=0,
        detections=(
            Detection(
                class_id=0,
                label="person",
                confidence=0.9,
                box=BoundingBox(10, 20, 90, 100),
                track_id=track_id,
            ),
        ),
    )
    return frame, result
