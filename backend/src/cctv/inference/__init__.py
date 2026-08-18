"""Object detection, tracking, and normalized inference results."""

from cctv.inference.base import ObjectDetector
from cctv.inference.factory import (
    DetectorConfig,
    DetectorConfigurationError,
    DetectorFactory,
    UnsupportedDetectorTypeError,
    built_in_detector_factory,
    create_detector,
)
from cctv.inference.models import (
    BoundingBox,
    Detection,
    DetectorMetadata,
    DetectorRunSummary,
    FrameDetections,
)
from cctv.inference.tracking import IoUTracker, TrackingSummary
from cctv.inference.yolo import (
    COCO_CLASS_NAMES,
    CpuYoloDetector,
    YoloError,
    YoloInferenceError,
    YoloModelLoadError,
    YoloRunSummary,
    load_class_names,
)

__all__ = [
    "COCO_CLASS_NAMES",
    "BoundingBox",
    "CpuYoloDetector",
    "Detection",
    "DetectorConfig",
    "DetectorConfigurationError",
    "DetectorFactory",
    "DetectorMetadata",
    "DetectorRunSummary",
    "FrameDetections",
    "IoUTracker",
    "ObjectDetector",
    "TrackingSummary",
    "UnsupportedDetectorTypeError",
    "YoloError",
    "YoloInferenceError",
    "YoloModelLoadError",
    "YoloRunSummary",
    "built_in_detector_factory",
    "create_detector",
    "load_class_names",
]
