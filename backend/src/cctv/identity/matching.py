"""Model-neutral identity similarity matching for detected video faces."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, sqrt
from typing import Protocol

from cctv.db import IdentityEmbeddingVector
from cctv.identity.face import DetectedFaceEmbedding, FaceBounds
from cctv.inference import FrameDetections
from cctv.media import DecodedFrame

logger = logging.getLogger(__name__)


class FaceMatchingError(RuntimeError):
    """Face matching cannot start with the supplied configuration or candidates."""


class FaceMatchStatus(StrEnum):
    """Final identity assignment state for a detected face."""

    MATCHED = "matched"
    UNKNOWN = "unknown"


class FaceMatchRejectionReason(StrEnum):
    """Why a best candidate was not accepted as the detected identity."""

    BELOW_THRESHOLD = "below_threshold"
    AMBIGUOUS = "ambiguous"


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

    status: FaceMatchStatus
    rejection_reason: FaceMatchRejectionReason | None
    best_candidate: IdentitySimilarity
    second_best_similarity: float | None
    similarity_threshold: float
    minimum_margin: float

    @property
    def matched(self) -> bool:
        return self.status is FaceMatchStatus.MATCHED

    @property
    def identity_id(self) -> str | None:
        return self.best_candidate.identity_id if self.matched else None

    @property
    def external_id(self) -> str | None:
        return self.best_candidate.external_id if self.matched else None

    @property
    def display_name(self) -> str | None:
        return self.best_candidate.display_name if self.matched else None


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
    track_cache_enabled: bool = False
    tracked_person_detections: int = 0
    face_analysis_attempts: int = 0
    cache_hits: int = 0
    cached_tracks: int = 0


@dataclass(frozen=True, slots=True)
class _TrackFaceMatchCacheEntry:
    analyzed_at_seconds: float
    decision: FaceMatchDecision | None


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
        if best.similarity < self.similarity_threshold:
            status = FaceMatchStatus.UNKNOWN
            rejection_reason = FaceMatchRejectionReason.BELOW_THRESHOLD
        elif not margin_is_sufficient:
            status = FaceMatchStatus.UNKNOWN
            rejection_reason = FaceMatchRejectionReason.AMBIGUOUS
        else:
            status = FaceMatchStatus.MATCHED
            rejection_reason = None
        return FaceMatchDecision(
            status=status,
            rejection_reason=rejection_reason,
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
        unknown_retry_seconds: float = 2.0,
        observation_sink: Callable[[FaceMatchObservation, int | None], None] | None = None,
    ) -> None:
        if not isfinite(unknown_retry_seconds) or unknown_retry_seconds <= 0:
            raise ValueError("unknown_retry_seconds must be a finite number greater than zero")
        self.extractor = extractor
        self.matcher = matcher
        self.source_name = source_name
        self.unknown_retry_seconds = float(unknown_retry_seconds)
        self.observation_sink = observation_sink
        self.processed_frames = 0
        self.frames_with_faces = 0
        self.detected_faces = 0
        self.matched_faces = 0
        self.unknown_faces = 0
        self.best_similarity: float | None = None
        self.matched_identity_counts: defaultdict[str, int] = defaultdict(int)
        self.last_observations: tuple[FaceMatchObservation, ...] = ()
        self.tracked_person_detections = 0
        self.face_analysis_attempts = 0
        self.cache_hits = 0
        self._track_cache_enabled = False
        self._track_cache: dict[int, _TrackFaceMatchCacheEntry] = {}

    def __call__(self, frame: DecodedFrame) -> None:
        """Match all faces without object-track caching."""
        self.processed_frames += 1
        self.face_analysis_attempts += 1
        detected_faces = self.extractor.extract_many(
            frame.image,
            source_name=f"{self.source_name}:sample-{frame.sample_index}",
        )
        if detected_faces:
            self.frames_with_faces += 1
        self.last_observations = self._evaluate_faces(frame, detected_faces)

    def process_tracked(
        self,
        frame: DecodedFrame,
        result: FrameDetections,
        *,
        active_track_ids: Collection[int],
    ) -> None:
        """Match person tracks once and reuse accepted results while each track is active."""
        if result.source_index != frame.source_index or result.sample_index != frame.sample_index:
            raise ValueError("tracked detections must belong to the decoded frame")
        self.processed_frames += 1
        self._track_cache_enabled = True
        active_ids = set(active_track_ids)
        for cached_track_id in tuple(self._track_cache):
            if cached_track_id not in active_ids:
                del self._track_cache[cached_track_id]

        observations: list[FaceMatchObservation] = []
        frame_has_face = False
        person_detections = tuple(
            detection
            for detection in result.detections
            if detection.label.casefold() == "person" and detection.track_id is not None
        )
        self.tracked_person_detections += len(person_detections)
        for detection in person_detections:
            track_id = detection.track_id
            if track_id is None:
                continue
            cached = self._track_cache.get(track_id)
            if cached is not None and not self._should_reanalyze(
                cached,
                timestamp_seconds=frame.timestamp_seconds,
            ):
                self.cache_hits += 1
                self._log_cache_hit(frame, track_id, cached)
                continue

            self.face_analysis_attempts += 1
            x1 = max(0, min(detection.box.x1, frame.image.shape[1]))
            y1 = max(0, min(detection.box.y1, frame.image.shape[0]))
            x2 = max(x1, min(detection.box.x2, frame.image.shape[1]))
            y2 = max(y1, min(detection.box.y2, frame.image.shape[0]))
            if x2 <= x1 or y2 <= y1:
                detected_faces: tuple[DetectedFaceEmbedding, ...] = ()
            else:
                detected_faces = self.extractor.extract_many(
                    frame.image[y1:y2, x1:x2],
                    source_name=(
                        f"{self.source_name}:sample-{frame.sample_index}:track-{track_id}"
                    ),
                )
            if not detected_faces:
                self._track_cache[track_id] = _TrackFaceMatchCacheEntry(
                    analyzed_at_seconds=frame.timestamp_seconds,
                    decision=None,
                )
                logger.debug(
                    "Tracked person face was not detected",
                    extra={
                        "event": "tracked_person_face_not_detected",
                        "source_name": self.source_name,
                        "source_index": frame.source_index,
                        "sample_index": frame.sample_index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "track_id": track_id,
                        "retry_after_seconds": self.unknown_retry_seconds,
                    },
                )
                continue

            frame_has_face = True
            selected_face = max(
                detected_faces,
                key=lambda face: (
                    face.bounds.width * face.bounds.height,
                    face.detection_confidence,
                ),
            )
            frame_face = DetectedFaceEmbedding(
                bounds=FaceBounds(
                    x=selected_face.bounds.x + x1,
                    y=selected_face.bounds.y + y1,
                    width=selected_face.bounds.width,
                    height=selected_face.bounds.height,
                ),
                vector=selected_face.vector,
                detection_confidence=selected_face.detection_confidence,
            )
            track_observations = self._evaluate_faces(
                frame,
                (frame_face,),
                track_id=track_id,
            )
            observations.extend(track_observations)
            self._track_cache[track_id] = _TrackFaceMatchCacheEntry(
                analyzed_at_seconds=frame.timestamp_seconds,
                decision=track_observations[0].decision,
            )
        if frame_has_face:
            self.frames_with_faces += 1
        self.last_observations = tuple(observations)

    def _should_reanalyze(
        self,
        cached: _TrackFaceMatchCacheEntry,
        *,
        timestamp_seconds: float,
    ) -> bool:
        if cached.decision is not None and cached.decision.matched:
            return False
        return timestamp_seconds - cached.analyzed_at_seconds >= self.unknown_retry_seconds

    def get_cached_decision(self, track_id: int) -> FaceMatchDecision | None:
        """Return the latest recognition decision for an active track, if a face was found."""
        cached = self._track_cache.get(track_id)
        return cached.decision if cached is not None else None

    def _evaluate_faces(
        self,
        frame: DecodedFrame,
        detected_faces: Sequence[DetectedFaceEmbedding],
        *,
        track_id: int | None = None,
    ) -> tuple[FaceMatchObservation, ...]:
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
                    "track_id": track_id,
                    "face_bounds": {
                        "x": face.bounds.x,
                        "y": face.bounds.y,
                        "width": face.bounds.width,
                        "height": face.bounds.height,
                    },
                    "detection_confidence": face.detection_confidence,
                    "match_status": decision.status,
                    "rejection_reason": decision.rejection_reason,
                    "matched": decision.matched,
                    "identity_id": decision.identity_id,
                    "external_id": decision.external_id,
                    "display_name": decision.display_name,
                    "best_candidate_identity_id": decision.best_candidate.identity_id,
                    "best_candidate_external_id": decision.best_candidate.external_id,
                    "best_similarity": similarity,
                    "second_best_similarity": decision.second_best_similarity,
                    "similarity_threshold": decision.similarity_threshold,
                    "minimum_margin": decision.minimum_margin,
                },
            )
            if self.observation_sink is not None:
                self.observation_sink(observation, track_id)
        return tuple(observations)

    def _log_cache_hit(
        self,
        frame: DecodedFrame,
        track_id: int,
        cached: _TrackFaceMatchCacheEntry,
    ) -> None:
        decision = cached.decision
        logger.debug(
            "Tracked face match reused from cache",
            extra={
                "event": "tracked_face_match_cache_hit",
                "source_name": self.source_name,
                "source_index": frame.source_index,
                "sample_index": frame.sample_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "track_id": track_id,
                "cache_age_seconds": round(
                    frame.timestamp_seconds - cached.analyzed_at_seconds,
                    6,
                ),
                "match_status": (
                    decision.status if decision is not None else FaceMatchStatus.UNKNOWN
                ),
                "rejection_reason": (
                    decision.rejection_reason if decision is not None else "face_not_detected"
                ),
                "identity_id": decision.identity_id if decision is not None else None,
                "external_id": decision.external_id if decision is not None else None,
                "display_name": decision.display_name if decision is not None else None,
            },
        )

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
            track_cache_enabled=self._track_cache_enabled,
            tracked_person_detections=self.tracked_person_detections,
            face_analysis_attempts=self.face_analysis_attempts,
            cache_hits=self.cache_hits,
            cached_tracks=len(self._track_cache),
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
    "FaceMatchRejectionReason",
    "FaceMatchStatus",
    "FaceMatchingConsumer",
    "FaceMatchingError",
    "FaceMatchingRunSummary",
    "FrameFaceExtractor",
    "IdentitySimilarity",
]
