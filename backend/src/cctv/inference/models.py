"""Detector-independent object-detection contracts shared by the pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel coordinates in the original frame, using an exclusive lower-right edge."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector-independent object detection."""

    class_id: int
    label: str
    confidence: float
    box: BoundingBox
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class FrameDetections:
    """Detection result for one sampled video frame."""

    source_index: int
    sample_index: int
    timestamp_seconds: float
    frame_width: int
    frame_height: int
    inference_seconds: float
    detections: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class DetectorMetadata:
    """Stable model identity and runtime configuration for persistence."""

    detector_type: str
    model_path: Path
    model_name: str
    model_sha256: str
    device: str
    input_size: int
    confidence_threshold: float
    nms_threshold: float


@dataclass(frozen=True, slots=True)
class DetectorRunSummary:
    """Bounded aggregate statistics shared by detector implementations."""

    detector_type: str
    model_path: Path
    model_name: str
    model_sha256: str
    device: str
    input_size: int
    confidence_threshold: float
    nms_threshold: float
    processed_frames: int
    total_detections: int
    total_inference_seconds: float
    average_inference_seconds: float


__all__ = [
    "BoundingBox",
    "Detection",
    "DetectorMetadata",
    "DetectorRunSummary",
    "FrameDetections",
]
