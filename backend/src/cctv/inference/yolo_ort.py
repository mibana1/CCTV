"""Strict ONNX Runtime CUDA inference with explicit, observable CPU fallback."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import file_digest
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from cctv.inference.models import DetectorMetadata, FrameDetections
from cctv.inference.yolo import (
    COCO_CLASS_NAMES,
    CpuYoloDetector,
    YoloInferenceError,
    YoloModelLoadError,
)
from cctv.media import DecodedFrame

logger = logging.getLogger(__name__)

CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"


class YoloProviderUnavailableError(RuntimeError):
    """Raised when strict CUDA inference was requested but cannot be initialized."""


class _OrtInput(Protocol):
    name: str


class _OrtSession(Protocol):
    def get_inputs(self) -> Sequence[_OrtInput]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(self, output_names: None, inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]: ...


class OrtCudaYoloDetector(CpuYoloDetector):
    """Run the existing YOLO pre/post-processing around a strict CUDA ORT session."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        session: _OrtSession,
        runtime_version: str,
        class_names: Sequence[str] = COCO_CLASS_NAMES,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> None:
        if input_size < 32:
            raise ValueError("input_size must be at least 32 pixels")
        if not isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be greater than 0 and at most 1")
        if not isfinite(nms_threshold) or not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        names = tuple(name.strip() for name in class_names if name.strip())
        if not names:
            raise ValueError("class_names must contain at least one non-empty label")

        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            raise YoloModelLoadError(f"YOLO ONNX model does not exist: {resolved_model}")
        if resolved_model.suffix.casefold() != ".onnx":
            raise YoloModelLoadError("CUDA YOLO currently requires an ONNX model")
        inputs = tuple(session.get_inputs())
        if len(inputs) != 1 or not inputs[0].name:
            raise YoloModelLoadError("YOLO ONNX model must expose exactly one named input")
        providers = tuple(session.get_providers())
        if CUDA_EXECUTION_PROVIDER not in providers:
            raise YoloProviderUnavailableError(
                "ONNX Runtime session did not activate CUDAExecutionProvider"
            )

        try:
            with resolved_model.open("rb") as model_file:
                model_sha256 = file_digest(model_file, "sha256").hexdigest()
        except OSError as error:
            raise YoloModelLoadError(
                f"YOLO ONNX model could not be read: {resolved_model}"
            ) from error

        self.model_path = resolved_model
        self.class_names = names
        self.model_sha256 = model_sha256
        self.input_size = input_size
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.requested_device = "cuda"
        self.execution_provider = CUDA_EXECUTION_PROVIDER
        self.cpu_fallback = False
        self.fallback_reason = None
        self.runtime_version = runtime_version
        self._session = session
        self._input_name = inputs[0].name
        self._processed_frames = 0
        self._total_detections = 0
        self._total_inference_seconds = 0.0

        logger.info(
            "CUDA YOLO detector initialized",
            extra={
                "event": "yolo_detector_initialized",
                "model_path": self.model_path,
                "model_sha256": self.model_sha256,
                "device": "cuda",
                "requested_device": "cuda",
                "execution_provider": self.execution_provider,
                "available_session_providers": providers,
                "cpu_fallback": False,
                "onnxruntime_version": runtime_version,
                "input_size": self.input_size,
                "class_count": len(self.class_names),
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
            },
        )

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            detector_type="yolo_onnx",
            model_path=self.model_path,
            model_name=self.model_path.name,
            model_sha256=self.model_sha256,
            device="cuda",
            input_size=self.input_size,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            requested_device="cuda",
            execution_provider=self.execution_provider,
            cpu_fallback=False,
        )

    def analyze(self, frame: DecodedFrame) -> FrameDetections:
        if frame.image.ndim != 3 or frame.image.shape[2] != 3 or frame.image.size == 0:
            raise YoloInferenceError("YOLO input must be a non-empty BGR image")

        padded, scale, pad_x, pad_y = self._letterbox(frame.image)
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        started_at = perf_counter()
        try:
            raw_output = self._session.run(None, {self._input_name: blob})
        except Exception as error:
            logger.exception(
                "CUDA YOLO inference failed",
                extra={
                    "event": "yolo_inference_failed",
                    "source_index": frame.source_index,
                    "sample_index": frame.sample_index,
                    "execution_provider": self.execution_provider,
                    "error_type": type(error).__name__,
                },
            )
            raise YoloInferenceError("YOLO ONNX Runtime CUDA inference failed") from error

        detections = self._decode_output(
            raw_output,
            original_width=frame.image.shape[1],
            original_height=frame.image.shape[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        inference_seconds = perf_counter() - started_at
        self._processed_frames += 1
        self._total_detections += len(detections)
        self._total_inference_seconds += inference_seconds
        result = FrameDetections(
            source_index=frame.source_index,
            sample_index=frame.sample_index,
            timestamp_seconds=frame.timestamp_seconds,
            frame_width=frame.image.shape[1],
            frame_height=frame.image.shape[0],
            inference_seconds=round(inference_seconds, 6),
            detections=detections,
        )
        logger.debug(
            "CUDA YOLO frame analyzed",
            extra={
                "event": "yolo_frame_analyzed",
                "source_index": frame.source_index,
                "sample_index": frame.sample_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "inference_seconds": result.inference_seconds,
                "detection_count": len(detections),
                "execution_provider": self.execution_provider,
                "detections": [asdict(detection) for detection in detections],
            },
        )
        return result


def create_cuda_yolo_detector(
    model_path: str | Path,
    *,
    class_names: Sequence[str] = COCO_CLASS_NAMES,
    input_size: int = 640,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    allow_cpu_fallback: bool = False,
    cuda_device_id: int = 0,
    cuda_gpu_mem_limit_mb: int | None = None,
) -> CpuYoloDetector | OrtCudaYoloDetector:
    """Create a strict CUDA session or an explicitly marked OpenCV CPU fallback."""
    try:
        runtime = importlib.import_module("onnxruntime")
    except ImportError:
        return _fallback_or_raise(
            "onnxruntime-gpu is not installed",
            allow_cpu_fallback=allow_cpu_fallback,
            model_path=model_path,
            class_names=class_names,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )

    available_providers = tuple(runtime.get_available_providers())
    if CUDA_EXECUTION_PROVIDER not in available_providers:
        return _fallback_or_raise(
            "CUDAExecutionProvider is unavailable; "
            f"available providers: {', '.join(available_providers) or 'none'}",
            allow_cpu_fallback=allow_cpu_fallback,
            model_path=model_path,
            class_names=class_names,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )

    session_options = runtime.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {"device_id": str(cuda_device_id)}
    if cuda_gpu_mem_limit_mb is not None:
        provider_options["gpu_mem_limit"] = str(cuda_gpu_mem_limit_mb * 1024 * 1024)
    try:
        session = runtime.InferenceSession(
            str(Path(model_path).expanduser().resolve()),
            sess_options=session_options,
            providers=[(CUDA_EXECUTION_PROVIDER, provider_options)],
        )
        return OrtCudaYoloDetector(
            model_path,
            session=session,
            runtime_version=str(getattr(runtime, "__version__", "unknown")),
            class_names=class_names,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
    except Exception as error:  # noqa: BLE001 - isolate third-party provider load failures
        return _fallback_or_raise(
            f"CUDA session initialization failed ({type(error).__name__})",
            allow_cpu_fallback=allow_cpu_fallback,
            model_path=model_path,
            class_names=class_names,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            cause=error,
        )


def _fallback_or_raise(
    reason: str,
    *,
    allow_cpu_fallback: bool,
    model_path: str | Path,
    class_names: Sequence[str],
    input_size: int,
    confidence_threshold: float,
    nms_threshold: float,
    cause: Exception | None = None,
) -> CpuYoloDetector:
    if not allow_cpu_fallback:
        error = YoloProviderUnavailableError(
            f"CUDA inference was requested but unavailable: {reason}. "
            "Set CCTV_AI_ALLOW_CPU_FALLBACK=true only when degraded CPU operation is acceptable."
        )
        if cause is not None:
            raise error from cause
        raise error

    logger.warning(
        "CUDA inference unavailable; using explicit CPU fallback",
        extra={
            "event": "yolo_cpu_fallback_activated",
            "requested_device": "cuda",
            "effective_device": "cpu",
            "execution_provider": "OpenCVDNNCPU",
            "cpu_fallback": True,
            "fallback_reason": reason,
        },
    )
    return CpuYoloDetector(
        model_path,
        class_names=class_names,
        input_size=input_size,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        requested_device="cuda",
        cpu_fallback=True,
        fallback_reason=reason,
    )


__all__ = [
    "CPU_EXECUTION_PROVIDER",
    "CUDA_EXECUTION_PROVIDER",
    "OrtCudaYoloDetector",
    "YoloProviderUnavailableError",
    "create_cuda_yolo_detector",
]
