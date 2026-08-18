"""CPU-only YOLO inference through OpenCV's ONNX DNN runtime."""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import file_digest
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from cctv.media import DecodedFrame

logger = logging.getLogger(__name__)

COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class YoloError(RuntimeError):
    """Base error raised by the CPU YOLO detector."""


class YoloModelLoadError(YoloError):
    """Raised when model artifacts cannot be loaded."""


class YoloInferenceError(YoloError):
    """Raised when a frame cannot be inferred or decoded."""


class _DnnNetwork(Protocol):
    def setPreferableBackend(self, backend_id: int) -> None: ...

    def setInput(self, blob: NDArray[np.float32]) -> None: ...

    def forward(self) -> NDArray[np.float32] | Sequence[NDArray[np.float32]]: ...


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel coordinates in the original frame, using an exclusive lower-right edge."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True, slots=True)
class Detection:
    """One normalized object detection."""

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
class YoloRunSummary:
    """Bounded aggregate statistics for a detector instance."""

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


def load_class_names(path: str | Path | None) -> tuple[str, ...]:
    """Load one class name per line, or use the standard COCO labels."""
    if path is None:
        return COCO_CLASS_NAMES

    class_path = Path(path).expanduser().resolve()
    if not class_path.is_file():
        raise YoloModelLoadError(f"YOLO class names file does not exist: {class_path}")

    try:
        names = tuple(
            line.strip()
            for line in class_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    except OSError as error:
        raise YoloModelLoadError(
            f"YOLO class names file could not be read: {class_path}"
        ) from error

    if not names:
        raise YoloModelLoadError(f"YOLO class names file is empty: {class_path}")
    return names


class CpuYoloDetector:
    """Run a YOLO ONNX model on sampled frames with OpenCV's CPU graph engine.

    Standard Ultralytics YOLOv8/YOLO11 outputs (``4 + classes``) and YOLOv5
    outputs (``5 + classes``) are accepted. Model acquisition is intentionally
    outside this class so normal application startup never downloads artifacts.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
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

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise YoloModelLoadError(f"YOLO ONNX model does not exist: {self.model_path}")
        if self.model_path.suffix.casefold() != ".onnx":
            raise YoloModelLoadError("CPU YOLO currently requires an ONNX model")

        try:
            with self.model_path.open("rb") as model_file:
                model_sha256 = file_digest(model_file, "sha256").hexdigest()
            network: _DnnNetwork = cv2.dnn.readNetFromONNX(str(self.model_path))
            network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        except (cv2.error, OSError) as error:
            raise YoloModelLoadError(
                f"YOLO ONNX model could not be loaded: {self.model_path}"
            ) from error

        self.class_names = names
        self.model_sha256 = model_sha256
        self.input_size = input_size
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self._network = network
        self._processed_frames = 0
        self._total_detections = 0
        self._total_inference_seconds = 0.0

        logger.info(
            "CPU YOLO detector initialized",
            extra={
                "event": "yolo_detector_initialized",
                "model_path": self.model_path,
                "model_sha256": self.model_sha256,
                "device": "cpu",
                "input_size": self.input_size,
                "class_count": len(self.class_names),
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
            },
        )

    @property
    def summary(self) -> YoloRunSummary:
        """Return aggregate inference statistics without retaining frame images."""
        average = (
            self._total_inference_seconds / self._processed_frames
            if self._processed_frames
            else 0.0
        )
        return YoloRunSummary(
            model_path=self.model_path,
            model_name=self.model_path.name,
            model_sha256=self.model_sha256,
            device="cpu",
            input_size=self.input_size,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            processed_frames=self._processed_frames,
            total_detections=self._total_detections,
            total_inference_seconds=round(self._total_inference_seconds, 6),
            average_inference_seconds=round(average, 6),
        )

    def __call__(self, frame: DecodedFrame) -> None:
        """Allow the detector to be attached directly as a frame consumer."""
        self.analyze(frame)

    def analyze(self, frame: DecodedFrame) -> FrameDetections:
        """Preprocess, infer, apply class-aware NMS, and normalize one frame."""
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
            self._network.setInput(blob)
            raw_output = self._network.forward()
        except cv2.error as error:
            logger.exception(
                "CPU YOLO inference failed",
                extra={
                    "event": "yolo_inference_failed",
                    "source_index": frame.source_index,
                    "sample_index": frame.sample_index,
                },
            )
            raise YoloInferenceError("YOLO ONNX inference failed") from error

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
            "CPU YOLO frame analyzed",
            extra={
                "event": "yolo_frame_analyzed",
                "source_index": frame.source_index,
                "sample_index": frame.sample_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "inference_seconds": result.inference_seconds,
                "detection_count": len(detections),
                "detections": [asdict(detection) for detection in detections],
            },
        )
        return result

    def _letterbox(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, min(self.input_size, round(width * scale)))
        resized_height = max(1, min(self.input_size, round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return padded, scale, pad_x, pad_y

    def _decode_output(
        self,
        output: NDArray[np.float32] | Sequence[NDArray[np.float32]],
        *,
        original_width: int,
        original_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> tuple[Detection, ...]:
        rows, has_objectness = self._prediction_rows(output)
        candidate_boxes: list[list[int]] = []
        candidate_scores: list[float] = []
        candidate_class_ids: list[int] = []

        for row in rows:
            class_scores = row[5:] if has_objectness else row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if has_objectness:
                confidence *= float(row[4])
            if not isfinite(confidence) or confidence < self.confidence_threshold:
                continue

            center_x, center_y, box_width, box_height = map(float, row[:4])
            coordinates = (center_x, center_y, box_width, box_height)
            if not all(isfinite(value) for value in coordinates):
                continue

            x1 = max(0.0, min(float(original_width), (center_x - box_width / 2 - pad_x) / scale))
            y1 = max(
                0.0,
                min(float(original_height), (center_y - box_height / 2 - pad_y) / scale),
            )
            x2 = max(0.0, min(float(original_width), (center_x + box_width / 2 - pad_x) / scale))
            y2 = max(
                0.0,
                min(float(original_height), (center_y + box_height / 2 - pad_y) / scale),
            )
            left = max(0, min(original_width - 1, round(x1)))
            top = max(0, min(original_height - 1, round(y1)))
            right = max(1, min(original_width, round(x2)))
            bottom = max(1, min(original_height, round(y2)))
            if right <= left or bottom <= top:
                continue

            candidate_boxes.append([left, top, right - left, bottom - top])
            candidate_scores.append(confidence)
            candidate_class_ids.append(class_id)

        selected_indices: list[int] = []
        for class_id in sorted(set(candidate_class_ids)):
            class_indices = [
                index
                for index, candidate_class_id in enumerate(candidate_class_ids)
                if candidate_class_id == class_id
            ]
            nms_indices = cv2.dnn.NMSBoxes(
                [candidate_boxes[index] for index in class_indices],
                [candidate_scores[index] for index in class_indices],
                self.confidence_threshold,
                self.nms_threshold,
            )
            selected_indices.extend(
                class_indices[int(local_index)]
                for local_index in np.asarray(nms_indices).reshape(-1)
            )

        selected_indices.sort(key=lambda index: candidate_scores[index], reverse=True)
        return tuple(
            Detection(
                class_id=candidate_class_ids[index],
                label=self.class_names[candidate_class_ids[index]],
                confidence=round(candidate_scores[index], 6),
                box=BoundingBox(
                    x1=candidate_boxes[index][0],
                    y1=candidate_boxes[index][1],
                    x2=candidate_boxes[index][0] + candidate_boxes[index][2],
                    y2=candidate_boxes[index][1] + candidate_boxes[index][3],
                ),
            )
            for index in selected_indices
        )

    def _prediction_rows(
        self, output: NDArray[np.float32] | Sequence[NDArray[np.float32]]
    ) -> tuple[NDArray[np.float32], bool]:
        arrays = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(arrays) != 1:
            raise YoloInferenceError(
                "YOLO model must expose one prediction output; export without embedded NMS"
            )

        predictions = np.asarray(arrays[0], dtype=np.float32)
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise YoloInferenceError(
                f"unsupported YOLO output shape: {tuple(np.asarray(arrays[0]).shape)}"
            )

        without_objectness = len(self.class_names) + 4
        with_objectness = len(self.class_names) + 5
        attribute_counts = {without_objectness, with_objectness}
        if predictions.shape[1] in attribute_counts:
            rows = predictions
        elif predictions.shape[0] in attribute_counts:
            rows = predictions.T
        else:
            raise YoloInferenceError(
                "YOLO output class count does not match the configured class names: "
                f"shape={tuple(predictions.shape)}, classes={len(self.class_names)}"
            )

        return rows, rows.shape[1] == with_objectness
