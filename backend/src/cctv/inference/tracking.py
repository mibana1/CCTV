"""Dependency-free, class-aware object tracking for sampled detection frames."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from math import isfinite

from cctv.inference.models import BoundingBox, Detection, FrameDetections


@dataclass(frozen=True, slots=True)
class TrackingSummary:
    """Aggregate counters for one tracker instance and analysis run."""

    processed_frames: int
    assigned_detections: int
    created_tracks: int
    matched_detections: int
    active_tracks: int


@dataclass(slots=True)
class _Track:
    track_id: int
    class_id: int
    box: BoundingBox
    last_timestamp_seconds: float
    missed_frames: int = 0


class IoUTracker:
    """Assign stable per-run IDs using class-aware intersection-over-union matching.

    The tracker is intentionally lightweight for the current CPU-only pipeline. Track
    IDs are unique only inside one tracker instance (and therefore one analysis run).
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 10,
        max_idle_seconds: float = 12.0,
        tracked_class_names: Collection[str] | None = None,
    ) -> None:
        if not isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be a finite number between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be zero or greater")
        if not isfinite(max_idle_seconds) or max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be a finite number greater than zero")
        normalized_class_names = (
            frozenset(name.strip().casefold() for name in tracked_class_names)
            if tracked_class_names is not None
            else None
        )
        if normalized_class_names is not None and (
            not normalized_class_names or "" in normalized_class_names
        ):
            raise ValueError("tracked_class_names must contain non-empty class names")

        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max_missed_frames
        self.max_idle_seconds = float(max_idle_seconds)
        self.tracked_class_names = normalized_class_names
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._processed_frames = 0
        self._assigned_detections = 0
        self._created_tracks = 0
        self._matched_detections = 0
        self._last_sample_index: int | None = None
        self._last_timestamp_seconds: float | None = None

    @property
    def summary(self) -> TrackingSummary:
        """Return immutable counters suitable for logs and worker JSON output."""
        return TrackingSummary(
            processed_frames=self._processed_frames,
            assigned_detections=self._assigned_detections,
            created_tracks=self._created_tracks,
            matched_detections=self._matched_detections,
            active_tracks=len(self._tracks),
        )

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        """Return tracker IDs that remain eligible for a future match."""
        return tuple(sorted(self._tracks))

    def update(self, result: FrameDetections) -> FrameDetections:
        """Match one chronologically ordered frame and return detections with IDs."""
        self._validate_order(result)
        self._expire_idle_tracks(result.timestamp_seconds)

        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(result.detections):
            if not self._is_trackable(detection):
                continue
            for track_id, track in self._tracks.items():
                if track.class_id != detection.class_id:
                    continue
                overlap = _intersection_over_union(track.box, detection.box)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, detection_index))

        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, detection_index in sorted(
            candidates,
            key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
        ):
            if track_id in matched_track_ids or detection_index in matched_detection_indexes:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(detection_index)
            assignments[detection_index] = track_id

        for track_id, track in tuple(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        tracked_detections: list[Detection] = []
        for detection_index, detection in enumerate(result.detections):
            if not self._is_trackable(detection):
                tracked_detections.append(replace(detection, track_id=None))
                continue
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    class_id=detection.class_id,
                    box=detection.box,
                    last_timestamp_seconds=result.timestamp_seconds,
                )
                self._created_tracks += 1
            else:
                track = self._tracks[track_id]
                track.box = detection.box
                track.last_timestamp_seconds = result.timestamp_seconds
                track.missed_frames = 0
                self._matched_detections += 1
            tracked_detections.append(replace(detection, track_id=track_id))

        self._processed_frames += 1
        self._assigned_detections += sum(
            detection.track_id is not None for detection in tracked_detections
        )
        self._last_sample_index = result.sample_index
        self._last_timestamp_seconds = result.timestamp_seconds
        return replace(result, detections=tuple(tracked_detections))

    def _is_trackable(self, detection: Detection) -> bool:
        return (
            self.tracked_class_names is None
            or detection.label.strip().casefold() in self.tracked_class_names
        )

    def _validate_order(self, result: FrameDetections) -> None:
        if self._last_sample_index is not None and result.sample_index <= self._last_sample_index:
            raise ValueError("tracking frames must have strictly increasing sample indexes")
        if (
            self._last_timestamp_seconds is not None
            and result.timestamp_seconds < self._last_timestamp_seconds
        ):
            raise ValueError("tracking frames must have non-decreasing timestamps")

    def _expire_idle_tracks(self, timestamp_seconds: float) -> None:
        for track_id, track in tuple(self._tracks.items()):
            if timestamp_seconds - track.last_timestamp_seconds > self.max_idle_seconds:
                del self._tracks[track_id]


def _intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    return intersection_area / (left_area + right_area - intersection_area)


__all__ = ["IoUTracker", "TrackingSummary"]
