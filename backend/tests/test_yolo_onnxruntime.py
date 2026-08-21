from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from cctv.inference import (
    COCO_CLASS_NAMES,
    CUDA_EXECUTION_PROVIDER,
    CpuYoloDetector,
    DetectorConfig,
    OrtCudaYoloDetector,
    YoloProviderUnavailableError,
    create_detector,
)
from cctv.media import DecodedFrame


class FakeOpenCvNetwork:
    def setPreferableBackend(self, backend_id: int) -> None:
        self.backend_id = backend_id

    def setInput(self, blob: np.ndarray) -> None:
        self.input_blob = blob

    def forward(self) -> np.ndarray:
        return np.zeros((1, len(COCO_CLASS_NAMES) + 4, 1), dtype=np.float32)


class FakeSessionOptions:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, name: str, value: str) -> None:
        self.entries[name] = value


class FakeCudaSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.inputs: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> tuple[SimpleNamespace, ...]:
        return (SimpleNamespace(name="images"),)

    def get_providers(self) -> tuple[str, ...]:
        return (CUDA_EXECUTION_PROVIDER,)

    def run(self, output_names: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names is None
        self.inputs = inputs
        return [self.output]


class FakeRuntime:
    __version__ = "1.26.0-test"

    def __init__(
        self,
        providers: tuple[str, ...],
        *,
        session: FakeCudaSession | None = None,
        session_error: Exception | None = None,
    ) -> None:
        self.providers = providers
        self.session = session
        self.session_error = session_error
        self.options: FakeSessionOptions | None = None
        self.provider_argument: Any = None

    def get_available_providers(self) -> tuple[str, ...]:
        return self.providers

    def SessionOptions(self) -> FakeSessionOptions:
        self.options = FakeSessionOptions()
        return self.options

    def InferenceSession(
        self,
        model_path: str,
        *,
        sess_options: FakeSessionOptions,
        providers: Any,
    ) -> FakeCudaSession:
        assert model_path.endswith("model.onnx")
        assert sess_options is self.options
        self.provider_argument = providers
        if self.session_error is not None:
            raise self.session_error
        assert self.session is not None
        return self.session


def detector_config(model_path: Path, **options: object) -> DetectorConfig:
    return DetectorConfig(
        detector_type="yolo_onnx",
        model_path=model_path,
        class_names_path=None,
        device="cuda",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        options=options,
    )


def install_runtime(monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime) -> None:
    monkeypatch.setattr(
        "cctv.inference.yolo_ort.importlib.import_module",
        lambda name: runtime if name == "onnxruntime" else None,
    )


def test_cuda_request_fails_closed_when_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    install_runtime(monkeypatch, FakeRuntime(("CPUExecutionProvider",)))

    with pytest.raises(YoloProviderUnavailableError, match="available providers"):
        create_detector(detector_config(model_path))


def test_cuda_request_uses_observable_cpu_fallback_only_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    install_runtime(monkeypatch, FakeRuntime(("CPUExecutionProvider",)))
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: FakeOpenCvNetwork())

    detector = create_detector(detector_config(model_path, allow_cpu_fallback=True))

    assert isinstance(detector, CpuYoloDetector)
    assert detector.metadata.requested_device == "cuda"
    assert detector.metadata.device == "cpu"
    assert detector.metadata.execution_provider == "OpenCVDNNCPU"
    assert detector.metadata.cpu_fallback is True
    assert "CUDAExecutionProvider is unavailable" in (detector.metadata.fallback_reason or "")


def test_cuda_session_disables_implicit_cpu_fallback_and_runs_yolo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    output = np.zeros((1, len(COCO_CLASS_NAMES) + 4, 1), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 100, 100]
    output[0, 4, 0] = 0.9
    session = FakeCudaSession(output)
    runtime = FakeRuntime((CUDA_EXECUTION_PROVIDER, "CPUExecutionProvider"), session=session)
    install_runtime(monkeypatch, runtime)

    detector = create_detector(
        detector_config(
            model_path,
            allow_cpu_fallback=False,
            cuda_device_id=2,
            cuda_gpu_mem_limit_mb=1024,
        )
    )
    result = detector.analyze(
        DecodedFrame(
            source_index=0,
            sample_index=1,
            timestamp_seconds=0.5,
            image=np.zeros((640, 640, 3), dtype=np.uint8),
        )
    )

    assert isinstance(detector, OrtCudaYoloDetector)
    assert detector.metadata.device == "cuda"
    assert detector.metadata.requested_device == "cuda"
    assert detector.metadata.execution_provider == CUDA_EXECUTION_PROVIDER
    assert detector.metadata.cpu_fallback is False
    assert runtime.options is not None
    assert runtime.options.entries == {"session.disable_cpu_ep_fallback": "1"}
    assert runtime.provider_argument == [
        (
            CUDA_EXECUTION_PROVIDER,
            {"device_id": "2", "gpu_mem_limit": str(1024 * 1024 * 1024)},
        )
    ]
    assert session.inputs is not None
    assert session.inputs["images"].shape == (1, 3, 640, 640)
    assert result.detections[0].label == "person"
    assert detector.summary.execution_provider == CUDA_EXECUTION_PROVIDER


def test_cuda_session_initialization_error_can_activate_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    runtime = FakeRuntime(
        (CUDA_EXECUTION_PROVIDER,),
        session_error=RuntimeError("missing CUDA library"),
    )
    install_runtime(monkeypatch, runtime)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: FakeOpenCvNetwork())

    detector = create_detector(detector_config(model_path, allow_cpu_fallback=True))

    assert detector.metadata.device == "cpu"
    assert detector.metadata.cpu_fallback is True
    assert detector.metadata.fallback_reason == "CUDA session initialization failed (RuntimeError)"
