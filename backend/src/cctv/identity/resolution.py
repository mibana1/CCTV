"""Resolve short-lived object tracks into stable people within one analysis run."""

from __future__ import annotations

import logging
from math import hypot, isfinite

from cctv.db import (
    FaceMatchEventRecord,
    PersonInstanceRecord,
    PersonInstanceRepository,
)

logger = logging.getLogger(__name__)


class TrackIdentityResolver:
    """Combine tracker IDs using accepted identity matches and conservative continuity."""

    def __init__(
        self,
        repository: PersonInstanceRepository,
        *,
        max_gap_seconds: float = 5.0,
        minimum_candidate_similarity: float = 0.35,
        max_center_distance_ratio: float = 6.0,
    ) -> None:
        if not isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be a finite number greater than zero")
        if (
            not isfinite(minimum_candidate_similarity)
            or not -1 <= minimum_candidate_similarity <= 1
        ):
            raise ValueError("minimum_candidate_similarity must be between -1 and 1")
        if not isfinite(max_center_distance_ratio) or max_center_distance_ratio <= 0:
            raise ValueError(
                "max_center_distance_ratio must be a finite number greater than zero"
            )
        self.repository = repository
        self.max_gap_seconds = float(max_gap_seconds)
        self.minimum_candidate_similarity = float(minimum_candidate_similarity)
        self.max_center_distance_ratio = float(max_center_distance_ratio)

    def resolve(self, event: FaceMatchEventRecord) -> PersonInstanceRecord | None:
        """Associate one persisted face decision with a session-scoped person."""
        if event.track_id is None:
            return None
        linked = self.repository.get_by_track(event.analysis_run_id, event.track_id)
        if event.match_status == "matched":
            return self._resolve_matched(event, linked)
        if linked is not None:
            return self.repository.apply_event(
                linked.id,
                event,
                linked_by="unknown_observation",
                confidence=event.best_similarity,
            )

        candidate = self._best_stitch_candidate(event)
        if candidate is not None:
            resolved = self.repository.apply_event(
                candidate.id,
                event,
                linked_by="candidate_stitch",
                confidence=event.best_similarity,
            )
            self._log_resolution(event, resolved, "candidate_stitch")
            return resolved
        resolved = self.repository.create_from_event(
            event,
            linked_by="unknown_observation",
            confidence=event.best_similarity,
        )
        self._log_resolution(event, resolved, "unknown_observation")
        return resolved

    def _resolve_matched(
        self,
        event: FaceMatchEventRecord,
        linked: PersonInstanceRecord | None,
    ) -> PersonInstanceRecord:
        if event.identity_id is None:
            raise ValueError("matched face events must include identity_id")
        identified = self.repository.get_by_identity(
            event.analysis_run_id,
            event.identity_id,
        )
        if linked is not None:
            target = linked
            if identified is not None and identified.id != linked.id:
                target = self.repository.merge_instances(identified.id, linked.id)
            resolved = self.repository.apply_event(
                target.id,
                event,
                linked_by="face_match",
                confidence=event.best_similarity,
            )
            self._log_resolution(event, resolved, "face_match")
            return resolved

        if identified is not None:
            stitch_candidate = self._best_stitch_candidate(
                event,
                excluded_ids={identified.id},
                unknown_only=True,
            )
            if stitch_candidate is not None:
                identified = self.repository.merge_instances(
                    identified.id,
                    stitch_candidate.id,
                )
            resolved = self.repository.apply_event(
                identified.id,
                event,
                linked_by="identity_match",
                confidence=event.best_similarity,
            )
            self._log_resolution(event, resolved, "identity_match")
            return resolved

        stitch_candidate = self._best_stitch_candidate(event, unknown_only=True)
        if stitch_candidate is not None:
            resolved = self.repository.apply_event(
                stitch_candidate.id,
                event,
                linked_by="candidate_stitch",
                confidence=event.best_similarity,
            )
            self._log_resolution(event, resolved, "candidate_stitch")
            return resolved
        resolved = self.repository.create_from_event(
            event,
            linked_by="face_match",
            confidence=event.best_similarity,
        )
        self._log_resolution(event, resolved, "face_match")
        return resolved

    def _best_stitch_candidate(
        self,
        event: FaceMatchEventRecord,
        *,
        excluded_ids: set[str] | None = None,
        unknown_only: bool = False,
    ) -> PersonInstanceRecord | None:
        excluded = excluded_ids or set()
        scored: list[tuple[float, float, PersonInstanceRecord]] = []
        candidates = self.repository.list_stitch_candidates(
            analysis_run_id=event.analysis_run_id,
            candidate_identity_id=event.best_candidate_identity_id,
            observed_at_seconds=event.source_timestamp_seconds,
            max_gap_seconds=self.max_gap_seconds,
        )
        for candidate in candidates:
            if candidate.id in excluded or (unknown_only and candidate.status != "unknown"):
                continue
            continuity = self._continuity_score(candidate, event)
            if continuity is None:
                continue
            scored.append((continuity, -candidate.best_similarity, candidate))
        return min(scored, default=(0.0, 0.0, None), key=lambda item: item[:2])[2]

    def _continuity_score(
        self,
        candidate: PersonInstanceRecord,
        event: FaceMatchEventRecord,
    ) -> float | None:
        gap = event.source_timestamp_seconds - candidate.last_seen_timestamp_seconds
        candidate_similarity_is_strong = (
            min(candidate.best_similarity, event.best_similarity)
            >= self.minimum_candidate_similarity
        )
        allowed_gap = (
            self.max_gap_seconds
            if candidate_similarity_is_strong
            else min(self.max_gap_seconds, 2.0)
        )
        if gap < 0 or gap > allowed_gap:
            return None
        candidate_x = candidate.last_face_x + candidate.last_face_width / 2
        candidate_y = candidate.last_face_y + candidate.last_face_height / 2
        event_x = event.face_x + event.face_width / 2
        event_y = event.face_y + event.face_height / 2
        scale = max(
            candidate.last_face_width,
            candidate.last_face_height,
            event.face_width,
            event.face_height,
        )
        distance_ratio = hypot(event_x - candidate_x, event_y - candidate_y) / scale
        allowed_distance_ratio = (
            self.max_center_distance_ratio
            if candidate_similarity_is_strong
            else self.max_center_distance_ratio * (2 / 3)
        )
        if distance_ratio > allowed_distance_ratio:
            return None
        return distance_ratio + gap / allowed_gap

    @staticmethod
    def _log_resolution(
        event: FaceMatchEventRecord,
        instance: PersonInstanceRecord,
        linked_by: str,
    ) -> None:
        logger.info(
            "Track identity resolved",
            extra={
                "event": "track_identity_resolved",
                "analysis_run_id": event.analysis_run_id,
                "sample_index": event.sample_index,
                "track_id": event.track_id,
                "person_instance_id": instance.id,
                "identity_id": instance.identity_id,
                "status": instance.status,
                "linked_by": linked_by,
                "track_count": instance.track_count,
            },
        )


__all__ = ["TrackIdentityResolver"]
