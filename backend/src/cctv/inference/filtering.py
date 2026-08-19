"""Detector-independent filtering for configured object classes."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace

from cctv.inference.models import FrameDetections


def filter_detections_by_class(
    result: FrameDetections,
    allowed_class_names: Collection[str],
) -> FrameDetections:
    """Keep configured classes while preserving frame metadata and detection order."""
    normalized = frozenset(name.strip().casefold() for name in allowed_class_names)
    if not normalized or "" in normalized:
        raise ValueError("allowed_class_names must contain non-empty class names")
    return replace(
        result,
        detections=tuple(
            detection
            for detection in result.detections
            if detection.label.strip().casefold() in normalized
        ),
    )


__all__ = ["filter_detections_by_class"]
