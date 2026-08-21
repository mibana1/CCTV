"""Registration-based construction of interchangeable object detectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cctv.inference.base import ObjectDetector

DetectorBuilder = Callable[["DetectorConfig"], ObjectDetector]


class DetectorConfigurationError(ValueError):
    """Raised when a detector type or its runtime configuration is invalid."""


class UnsupportedDetectorTypeError(DetectorConfigurationError):
    """Raised when no builder is registered for the requested detector type."""


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Common construction inputs plus optional adapter-specific values."""

    detector_type: str
    model_path: Path
    class_names_path: Path | None
    device: str
    input_size: int
    confidence_threshold: float
    nms_threshold: float
    options: Mapping[str, object] = field(default_factory=dict)


class DetectorFactory:
    """Resolve detector builders without coupling workers to implementations."""

    def __init__(self) -> None:
        self._builders: dict[str, DetectorBuilder] = {}

    @property
    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def register(self, detector_type: str, builder: DetectorBuilder) -> None:
        normalized = _normalize_type(detector_type)
        if normalized in self._builders:
            raise ValueError(f"duplicate detector type: {normalized}")
        self._builders[normalized] = builder

    def create(self, config: DetectorConfig) -> ObjectDetector:
        detector_type = _normalize_type(config.detector_type)
        try:
            builder = self._builders[detector_type]
        except KeyError as error:
            supported = ", ".join(self.supported_types)
            raise UnsupportedDetectorTypeError(
                f"unsupported detector type {detector_type!r}; registered types: {supported}"
            ) from error
        detector = builder(config)
        if not isinstance(detector, ObjectDetector):
            raise TypeError(
                f"builder for detector type {detector_type!r} did not return an ObjectDetector"
            )
        return detector


def built_in_detector_factory() -> DetectorFactory:
    """Return a fresh factory containing application-supported adapters."""
    factory = DetectorFactory()
    factory.register("yolo_onnx", _build_yolo_onnx)
    return factory


def create_detector(config: DetectorConfig) -> ObjectDetector:
    """Construct a detector from the built-in application registry."""
    return built_in_detector_factory().create(config)


def _build_yolo_onnx(config: DetectorConfig) -> ObjectDetector:
    from cctv.inference.yolo import CpuYoloDetector, load_class_names
    from cctv.inference.yolo_ort import create_cuda_yolo_detector

    device = config.device.casefold()
    common = {
        "class_names": load_class_names(config.class_names_path),
        "input_size": config.input_size,
        "confidence_threshold": config.confidence_threshold,
        "nms_threshold": config.nms_threshold,
    }
    if device == "cpu":
        return CpuYoloDetector(config.model_path, **common)
    if device == "cuda":
        return create_cuda_yolo_detector(
            config.model_path,
            **common,
            allow_cpu_fallback=_boolean_option(config, "allow_cpu_fallback", False),
            cuda_device_id=_integer_option(config, "cuda_device_id", 0, minimum=0),
            cuda_gpu_mem_limit_mb=_optional_integer_option(
                config,
                "cuda_gpu_mem_limit_mb",
                minimum=256,
            ),
        )
    raise DetectorConfigurationError("yolo_onnx device must be cpu or cuda")


def _boolean_option(config: DetectorConfig, name: str, default: bool) -> bool:
    value = config.options.get(name, default)
    if not isinstance(value, bool):
        raise DetectorConfigurationError(f"{name} must be true or false")
    return value


def _integer_option(
    config: DetectorConfig,
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = config.options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DetectorConfigurationError(f"{name} must be an integer of at least {minimum}")
    return value


def _optional_integer_option(
    config: DetectorConfig,
    name: str,
    *,
    minimum: int,
) -> int | None:
    value = config.options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DetectorConfigurationError(f"{name} must be an integer of at least {minimum}")
    return value


def _normalize_type(value: str) -> str:
    normalized = value.strip().casefold()
    if not normalized:
        raise DetectorConfigurationError("detector_type must not be empty")
    return normalized


__all__ = [
    "DetectorBuilder",
    "DetectorConfig",
    "DetectorConfigurationError",
    "DetectorFactory",
    "UnsupportedDetectorTypeError",
    "built_in_detector_factory",
    "create_detector",
]
