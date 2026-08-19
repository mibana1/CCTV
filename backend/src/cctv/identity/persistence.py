"""Persist face decisions and resolve tracker IDs into session-scoped people."""

from __future__ import annotations

from collections.abc import Callable

from cctv.core.settings import Settings
from cctv.db import (
    FaceMatchEventInput,
    FaceMatchRepository,
    PersonInstanceRepository,
)
from cctv.identity.matching import FaceMatchObservation
from cctv.identity.resolution import TrackIdentityResolver


def create_resolving_face_observation_sink(
    settings: Settings,
    *,
    analysis_run_id: str,
) -> Callable[[FaceMatchObservation, int | None], None]:
    """Build one synchronous sink that retains raw events and stable person links."""
    event_repository = FaceMatchRepository(settings.database_path)
    resolver = TrackIdentityResolver(
        PersonInstanceRepository(settings.database_path),
        max_gap_seconds=settings.face_identity_stitch_max_gap_seconds,
        minimum_candidate_similarity=settings.face_identity_stitch_min_similarity,
        max_center_distance_ratio=settings.face_identity_stitch_max_distance_ratio,
    )

    def persist(observation: FaceMatchObservation, track_id: int | None) -> None:
        decision = observation.decision
        best_candidate = decision.best_candidate
        event = event_repository.save_event(
            analysis_run_id,
            FaceMatchEventInput(
                source_index=observation.source_index,
                sample_index=observation.sample_index,
                source_timestamp_seconds=observation.timestamp_seconds,
                face_index=observation.face_index,
                track_id=track_id,
                face_x=observation.face.bounds.x,
                face_y=observation.face.bounds.y,
                face_width=observation.face.bounds.width,
                face_height=observation.face.bounds.height,
                detection_confidence=observation.face.detection_confidence,
                match_status=decision.status.value,
                rejection_reason=(
                    decision.rejection_reason.value
                    if decision.rejection_reason is not None
                    else None
                ),
                identity_id=decision.identity_id,
                external_id=decision.external_id,
                display_name=decision.display_name,
                best_candidate_identity_id=(
                    best_candidate.identity_id if best_candidate is not None else None
                ),
                best_candidate_external_id=(
                    best_candidate.external_id if best_candidate is not None else None
                ),
                best_similarity=decision.best_similarity,
                second_best_similarity=decision.second_best_similarity,
                similarity_threshold=decision.similarity_threshold,
                minimum_margin=decision.minimum_margin,
            ),
        )
        if best_candidate is not None:
            resolver.resolve(event)

    return persist


__all__ = ["create_resolving_face_observation_sink"]

