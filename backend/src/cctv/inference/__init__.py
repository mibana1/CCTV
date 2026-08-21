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
from cctv.inference.filtering import filter_detections_by_class
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
from cctv.inference.yolo_ort import (
    CUDA_EXECUTION_PROVIDER,
    OrtCudaYoloDetector,
    YoloProviderUnavailableError,
    create_cuda_yolo_detector,
)

__all__ = [
    "COCO_CLASS_NAMES",
    "CUDA_EXECUTION_PROVIDER",
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
    "OrtCudaYoloDetector",
    "TrackingSummary",
    "UnsupportedDetectorTypeError",
    "YoloError",
    "YoloInferenceError",
    "YoloModelLoadError",
    "YoloProviderUnavailableError",
    "YoloRunSummary",
    "built_in_detector_factory",
    "create_cuda_yolo_detector",
    "create_detector",
    "filter_detections_by_class",
    "load_class_names",
]
