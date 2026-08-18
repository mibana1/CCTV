from pathlib import Path

import cv2
import numpy as np
import pytest

from cctv.inference import (
    COCO_CLASS_NAMES,
    BoundingBox,
    CpuYoloDetector,
    ObjectDetector,
    YoloInferenceError,
    YoloModelLoadError,
    load_class_names,
)
from cctv.media import DecodedFrame


class FakeNetwork:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.backend_id: int | None = None
        self.target_id: int | None = None
        self.input_blob: np.ndarray | None = None

    def setPreferableBackend(self, backend_id: int) -> None:
        self.backend_id = backend_id

    def setPreferableTarget(self, target_id: int) -> None:
        self.target_id = target_id

    def setInput(self, blob: np.ndarray) -> None:
        self.input_blob = blob

    def forward(self) -> np.ndarray:
        return self.output


def create_frame(*, width: int = 640, height: int = 320) -> DecodedFrame:
    return DecodedFrame(
        source_index=8,
        sample_index=2,
        timestamp_seconds=1.5,
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def install_fake_network(
    monkeypatch: pytest.MonkeyPatch,
    output: np.ndarray,
) -> FakeNetwork:
    network = FakeNetwork(output)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: network)
    return network


def test_cpu_yolo_decodes_v8_output_with_class_aware_nms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test model placeholder")
    output = np.zeros((1, len(COCO_CLASS_NAMES) + 4, 3), dtype=np.float32)
    output[0, :4, 0] = [320, 320, 200, 100]
    output[0, 4, 0] = 0.9
    output[0, :4, 1] = [325, 320, 200, 100]
    output[0, 4, 1] = 0.7
    output[0, :4, 2] = [320, 320, 200, 100]
    output[0, 4 + 2, 2] = 0.8
    network = install_fake_network(monkeypatch, output)

    detector = CpuYoloDetector(model_path)
    result = detector.analyze(create_frame())

    assert network.backend_id == cv2.dnn.DNN_BACKEND_OPENCV
    assert network.target_id is None
    assert network.input_blob is not None
    assert network.input_blob.shape == (1, 3, 640, 640)
    assert [detection.label for detection in result.detections] == ["person", "car"]
    assert [detection.confidence for detection in result.detections] == pytest.approx([0.9, 0.8])
    assert result.detections[0].box == BoundingBox(x1=220, y1=110, x2=420, y2=210)
    assert result.source_index == 8
    assert result.sample_index == 2
    assert result.frame_width == 640
    assert result.frame_height == 320
    assert detector.summary.processed_frames == 1
    assert isinstance(detector, ObjectDetector)
    assert detector.metadata.detector_type == "yolo_onnx"
    assert detector.summary.detector_type == "yolo_onnx"
    assert detector.summary.total_detections == 2
    assert detector.summary.device == "cpu"
    assert detector.summary.model_name == "model.onnx"
    assert len(detector.summary.model_sha256) == 64


def test_cpu_yolo_decodes_v5_objectness_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(b"test model placeholder")
    output = np.array([[[16, 16, 10, 10, 0.5, 0.8]]], dtype=np.float32)
    install_fake_network(monkeypatch, output)
    detector = CpuYoloDetector(
        model_path,
        class_names=("vehicle",),
        input_size=32,
        confidence_threshold=0.25,
    )

    result = detector.analyze(create_frame(width=32, height=32))

    assert len(result.detections) == 1
    assert result.detections[0].label == "vehicle"
    assert result.detections[0].confidence == pytest.approx(0.4)
    assert result.detections[0].box == BoundingBox(x1=11, y1=11, x2=21, y2=21)


def test_cpu_yolo_rejects_output_that_does_not_match_class_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test model placeholder")
    install_fake_network(monkeypatch, np.zeros((1, 84, 10), dtype=np.float32))
    detector = CpuYoloDetector(model_path, class_names=("only-class",))

    with pytest.raises(YoloInferenceError, match="class count"):
        detector.analyze(create_frame())


def test_cpu_yolo_requires_a_local_onnx_model(tmp_path: Path) -> None:
    with pytest.raises(YoloModelLoadError, match="does not exist"):
        CpuYoloDetector(tmp_path / "missing.onnx")

    pytorch_model = tmp_path / "model.pt"
    pytorch_model.write_bytes(b"test model placeholder")
    with pytest.raises(YoloModelLoadError, match="requires an ONNX model"):
        CpuYoloDetector(pytorch_model)


def test_load_class_names_uses_coco_or_explicit_file(tmp_path: Path) -> None:
    assert load_class_names(None) is COCO_CLASS_NAMES

    class_path = tmp_path / "classes.txt"
    class_path.write_text("person\n\nvehicle\n", encoding="utf-8")
    assert load_class_names(class_path) == ("person", "vehicle")

    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(YoloModelLoadError, match="is empty"):
        load_class_names(empty_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_size", 31),
        ("confidence_threshold", 0),
        ("confidence_threshold", 1.1),
        ("nms_threshold", -0.1),
        ("nms_threshold", 1.1),
    ],
)
def test_cpu_yolo_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test model placeholder")
    install_fake_network(monkeypatch, np.zeros((1, 84, 1), dtype=np.float32))

    with pytest.raises(ValueError):
        CpuYoloDetector(model_path, **{field: value})
