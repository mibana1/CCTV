from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from cctv.inference import (
    BoundingBox,
    Detection,
    DetectorConfig,
    DetectorFactory,
    DetectorMetadata,
    DetectorRunSummary,
    FrameDetections,
    ObjectDetector,
    UnsupportedDetectorTypeError,
    built_in_detector_factory,
)
from cctv.media import DecodedFrame


class FakeDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._processed_frames = 0

    @property
    def metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            detector_type="custom",
            model_path=self.config.model_path,
            model_name=self.config.model_path.name,
            model_sha256="a" * 64,
            device=self.config.device,
            input_size=self.config.input_size,
            confidence_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.nms_threshold,
        )

    @property
    def summary(self) -> DetectorRunSummary:
        return DetectorRunSummary(
            **asdict(self.metadata),
            processed_frames=self._processed_frames,
            total_detections=self._processed_frames,
            total_inference_seconds=0,
            average_inference_seconds=0,
        )

    def analyze(self, frame: DecodedFrame) -> FrameDetections:
        self._processed_frames += 1
        return FrameDetections(
            source_index=frame.source_index,
            sample_index=frame.sample_index,
            timestamp_seconds=frame.timestamp_seconds,
            frame_width=frame.image.shape[1],
            frame_height=frame.image.shape[0],
            inference_seconds=0,
            detections=(Detection(0, "object", 0.9, BoundingBox(0, 0, 1, 1)),),
        )


def config(tmp_path: Path, detector_type: str = "custom") -> DetectorConfig:
    return DetectorConfig(
        detector_type=detector_type,
        model_path=tmp_path / "custom.model",
        class_names_path=None,
        device="cpu",
        input_size=320,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        options={"variant": "tiny"},
    )


def test_factory_accepts_custom_detector_without_worker_changes(tmp_path: Path) -> None:
    factory = DetectorFactory()
    factory.register("custom", FakeDetector)

    detector = factory.create(config(tmp_path, "CUSTOM"))
    result = detector.analyze(
        DecodedFrame(
            source_index=1,
            sample_index=0,
            timestamp_seconds=0,
            image=np.zeros((10, 20, 3), dtype=np.uint8),
        )
    )

    assert isinstance(detector, ObjectDetector)
    assert detector.metadata.detector_type == "custom"
    assert detector.config.options == {"variant": "tiny"}
    assert result.frame_width == 20
    assert result.detections[0].label == "object"


def test_factory_reports_supported_types_and_rejects_unknown(tmp_path: Path) -> None:
    factory = built_in_detector_factory()

    assert factory.supported_types == ("yolo_onnx",)
    with pytest.raises(UnsupportedDetectorTypeError, match="registered types: yolo_onnx"):
        factory.create(config(tmp_path, "unknown"))


def test_factory_rejects_duplicate_registration() -> None:
    factory = DetectorFactory()
    factory.register("custom", FakeDetector)

    with pytest.raises(ValueError, match="duplicate detector type"):
        factory.register("CUSTOM", FakeDetector)


def test_factory_rejects_builder_that_breaks_detector_contract(tmp_path: Path) -> None:
    factory = DetectorFactory()
    factory.register("bad", lambda _: object())  # type: ignore[arg-type,return-value]

    with pytest.raises(TypeError, match="did not return an ObjectDetector"):
        factory.create(config(tmp_path, "bad"))
