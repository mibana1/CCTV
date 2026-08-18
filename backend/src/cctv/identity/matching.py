"""Model-neutral identity similarity matching for detected video faces."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Protocol

from cctv.db import IdentityEmbeddingVector
from cctv.identity.face import DetectedFaceEmbedding
from cctv.media import DecodedFrame

logger = logging.getLogger(__name__)


class FaceMatchingError(RuntimeError):
    """Face matching cannot start with the supplied configuration or candidates."""


class FrameFaceExtractor(Protocol):
    """Adapter contract for video-frame face embedding models."""

    def extract_many(
        self,
        image: object,
        *,
        source_name: str = "frame",
    ) -> tuple[DetectedFaceEmbedding, ...]: ...


@dataclass(frozen=True, slots=True)
class IdentitySimilarity:
    """Best registered-photo similarity for one identity."""

    identity_id: str
    external_id: str | None
    display_name: str
    embedding_id: str
    similarity: float


@dataclass(frozen=True, slots=True)
class FaceMatchDecision:
    """One detected face's best candidate and conservative match decision."""

    matched: bool
    best_candidate: IdentitySimilarity
    second_best_similarity: float | None
    similarity_threshold: float
    minimum_margin: float

    @property
    def identity_id(self) -> str | None:
        return self.best_candidate.identity_id if self.matched else None


@dataclass(frozen=True, slots=True)
class FaceMatchObservation:
    """A detected video face paired with an identity match decision."""

    source_index: int
    sample_index: int
    timestamp_seconds: float
    face_index: int
    face: DetectedFaceEmbedding
    decision: FaceMatchDecision


@dataclass(frozen=True, slots=True)
class FaceMatchingRunSummary:
    """Privacy-safe aggregate result for one worker run."""

    candidate_identity_count: int
    candidate_embedding_count: int
    similarity_threshold: float
    minimum_margin: float
    processed_frames: int
    frames_with_faces: int
    detected_faces: int
    matched_faces: int
    unknown_faces: int
    best_similarity: float | None
    matched_identity_counts: dict[str, int]


class FaceIdentityMatcher:
    """Compare normalized face vectors and select an identity conservatively."""

    def __init__(
        self,
        candidates: Collection[IdentityEmbeddingVector],
        *,
        similarity_threshold: float = 0.45,
        minimum_margin: float = 0.05,
    ) -> None:
        if not isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be greater than 0 and at most 1")
        if not isfinite(minimum_margin) or not 0 <= minimum_margin <= 1:
            raise ValueError("minimum_margin must be between 0 and 1")
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise FaceMatchingError("no enabled identity embeddings are registered")
        dimensions = {len(candidate.vector) for candidate in self.candidates}
        if len(dimensions) != 1:
            raise FaceMatchingError("identity embedding candidates must have one dimension")
        self.dimension = dimensions.pop()
        self.similarity_threshold = float(similarity_threshold)
        self.minimum_margin = float(minimum_margin)
        self.identity_count = len({candidate.identity.id for candidate in self.candidates})

    def match(self, vector: Sequence[float]) -> FaceMatchDecision:
        """Return the best per-identity cosine score and an acceptance decision."""
        normalized = _normalize_vector(vector, expected_dimension=self.dimension)
        scores_by_identity: dict[str, IdentitySimilarity] = {}
        for candidate in self.candidates:
            similarity = sum(
                query_value * candidate_value
                for query_value, candidate_value in zip(normalized, candidate.vector, strict=True)
            )
            score = IdentitySimilarity(
                identity_id=candidate.identity.id,
                external_id=candidate.identity.external_id,
                display_name=candidate.identity.display_name,
                embedding_id=candidate.embedding.id,
                similarity=round(float(similarity), 6),
            )
            previous = scores_by_identity.get(score.identity_id)
            if previous is None or score.similarity > previous.similarity:
                scores_by_identity[score.identity_id] = score

        ranked = sorted(
            scores_by_identity.values(),
            key=lambda item: (-item.similarity, item.identity_id),
        )
        best = ranked[0]
        second_best_similarity = ranked[1].similarity if len(ranked) > 1 else None
        margin_is_sufficient = (
            second_best_similarity is None
            or best.similarity - second_best_similarity >= self.minimum_margin
        )
        return FaceMatchDecision(
            matched=best.similarity >= self.similarity_threshold and margin_is_sufficient,
            best_candidate=best,
            second_best_similarity=second_best_similarity,
            similarity_threshold=self.similarity_threshold,
            minimum_margin=self.minimum_margin,
        )


class FaceMatchingConsumer:
    """Detect and match every face in each sampled decoded frame."""

    def __init__(
        self,
        extractor: FrameFaceExtractor,
        matcher: FaceIdentityMatcher,
        *,
        source_name: str,
    ) -> None:
        self.extractor = extractor
        self.matcher = matcher
        self.source_name = source_name
        self.processed_frames = 0
        self.frames_with_faces = 0
        self.detected_faces = 0
        self.matched_faces = 0
        self.unknown_faces = 0
        self.best_similarity: float | None = None
        self.matched_identity_counts: defaultdict[str, int] = defaultdict(int)
        self.last_observations: tuple[FaceMatchObservation, ...] = ()

    def __call__(self, frame: DecodedFrame) -> None:
        self.processed_frames += 1
        detected_faces = self.extractor.extract_many(
            frame.image,
            source_name=f"{self.source_name}:sample-{frame.sample_index}",
        )
        if detected_faces:
            self.frames_with_faces += 1
        observations: list[FaceMatchObservation] = []
        for face_index, face in enumerate(detected_faces):
            decision = self.matcher.match(face.vector)
            observation = FaceMatchObservation(
                source_index=frame.source_index,
                sample_index=frame.sample_index,
                timestamp_seconds=frame.timestamp_seconds,
                face_index=face_index,
                face=face,
                decision=decision,
            )
            observations.append(observation)
            self.detected_faces += 1
            similarity = decision.best_candidate.similarity
            self.best_similarity = (
                similarity if self.best_similarity is None else max(self.best_similarity, similarity)
            )
            if decision.matched:
                self.matched_faces += 1
                self.matched_identity_counts[decision.best_candidate.identity_id] += 1
            else:
                self.unknown_faces += 1
            logger.info(
                "Video face similarity evaluated",
                extra={
                    "event": "video_face_similarity_evaluated",
                    "source_name": self.source_name,
                    "source_index": frame.source_index,
                    "sample_index": frame.sample_index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "face_index": face_index,
                    "face_bounds": {
                        "x": face.bounds.x,
                        "y": face.bounds.y,
                        "width": face.bounds.width,
                        "height": face.bounds.height,
                    },
                    "detection_confidence": face.detection_confidence,
                    "matched": decision.matched,
                    "identity_id": decision.identity_id,
                    "best_candidate_identity_id": decision.best_candidate.identity_id,
                    "best_candidate_external_id": decision.best_candidate.external_id,
                    "best_similarity": similarity,
                    "second_best_similarity": decision.second_best_similarity,
                    "similarity_threshold": decision.similarity_threshold,
                    "minimum_margin": decision.minimum_margin,
                },
            )
        self.last_observations = tuple(observations)

    @property
    def summary(self) -> FaceMatchingRunSummary:
        return FaceMatchingRunSummary(
            candidate_identity_count=self.matcher.identity_count,
            candidate_embedding_count=len(self.matcher.candidates),
            similarity_threshold=self.matcher.similarity_threshold,
            minimum_margin=self.matcher.minimum_margin,
            processed_frames=self.processed_frames,
            frames_with_faces=self.frames_with_faces,
            detected_faces=self.detected_faces,
            matched_faces=self.matched_faces,
            unknown_faces=self.unknown_faces,
            best_similarity=self.best_similarity,
            matched_identity_counts=dict(sorted(self.matched_identity_counts.items())),
        )


def _normalize_vector(values: Sequence[float], *, expected_dimension: int) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != expected_dimension:
        raise FaceMatchingError(
            f"face embedding dimension must be {expected_dimension}; received {len(vector)}"
        )
    if not all(isfinite(value) for value in vector):
        raise FaceMatchingError("face embedding must contain only finite values")
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude <= 0:
        raise FaceMatchingError("face embedding must have a non-zero magnitude")
    return tuple(value / magnitude for value in vector)


__all__ = [
    "FaceIdentityMatcher",
    "FaceMatchDecision",
    "FaceMatchObservation",
    "FaceMatchingConsumer",
    "FaceMatchingError",
    "FaceMatchingRunSummary",
    "FrameFaceExtractor",
    "IdentitySimilarity",
]
