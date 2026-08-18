"""OpenCV YuNet and SFace adapters for local face embedding extraction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

SFACE_MODEL_NAME = "opencv_sface"
SFACE_MODEL_VERSION = "2021dec"
SFACE_EMBEDDING_DIMENSION = 128
YUNET_MODEL_NAME = "opencv_yunet"
YUNET_MODEL_VERSION = "2026may"


class FaceEmbeddingError(RuntimeError):
    """Base error for model loading, photo decoding, and feature extraction."""


class FaceModelLoadError(FaceEmbeddingError):
    """A configured YuNet or SFace ONNX file cannot be loaded."""


class FacePhotoError(FaceEmbeddingError):
    """A registration photo cannot produce exactly one valid face feature."""


@dataclass(frozen=True, slots=True)
class FaceEmbeddingModelMetadata:
    detector_name: str = YUNET_MODEL_NAME
    detector_version: str = YUNET_MODEL_VERSION
    model_name: str = SFACE_MODEL_NAME
    model_version: str = SFACE_MODEL_VERSION
    dimension: int = SFACE_EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class ExtractedFaceEmbedding:
    """One SFace vector and its YuNet detection confidence."""

    vector: tuple[float, ...]
    detection_confidence: float


@dataclass(frozen=True, slots=True)
class FaceBounds:
    """Pixel bounds returned by YuNet for one detected face."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DetectedFaceEmbedding:
    """One face detected in a video frame and its SFace vector."""

    bounds: FaceBounds
    vector: tuple[float, ...]
    detection_confidence: float


class OpenCvSFaceExtractor:
    """Detect, align, and embed faces with OpenCV YuNet and SFace."""

    metadata = FaceEmbeddingModelMetadata()

    def __init__(
        self,
        detection_model_path: str | Path,
        recognition_model_path: str | Path,
        *,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5_000,
        max_input_dimension: int = 960,
    ) -> None:
        if not isfinite(score_threshold) or not 0 < score_threshold <= 1:
            raise ValueError("score_threshold must be greater than 0 and at most 1")
        if not isfinite(nms_threshold) or not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 320 <= max_input_dimension <= 4_096:
            raise ValueError("max_input_dimension must be between 320 and 4096")

        self.detection_model_path = _required_model_file(
            detection_model_path,
            "YuNet detection",
        )
        self.recognition_model_path = _required_model_file(
            recognition_model_path,
            "SFace recognition",
        )
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = top_k
        self.max_input_dimension = max_input_dimension

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.detection_model_path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(
                str(self.recognition_model_path),
                "",
            )
        except cv2.error as error:
            raise FaceModelLoadError("YuNet or SFace model could not be loaded") from error

    def extract_file(self, photo_path: str | Path) -> ExtractedFaceEmbedding:
        """Decode one local image without relying on platform path encoding."""
        path = Path(photo_path).expanduser().resolve()
        if not path.is_file():
            raise FacePhotoError(f"registration photo does not exist: {path.name}")
        try:
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        except OSError as error:
            raise FacePhotoError(f"registration photo could not be read: {path.name}") from error
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise FacePhotoError(f"registration photo is not a decodable image: {path.name}")
        return self.extract(image, source_name=path.name)

    def extract(
        self,
        image: NDArray[np.uint8],
        *,
        source_name: str = "image",
    ) -> ExtractedFaceEmbedding:
        """Require one face, align it from five landmarks, and return its feature."""
        detected_faces = self.extract_many(image, source_name=source_name)
        face_count = len(detected_faces)
        if face_count != 1:
            raise FacePhotoError(
                f"registration photo must contain exactly one face; found {face_count}: "
                f"{source_name}"
            )
        detected = detected_faces[0]
        return ExtractedFaceEmbedding(
            vector=detected.vector,
            detection_confidence=detected.detection_confidence,
        )

    def extract_many(
        self,
        image: NDArray[np.uint8],
        *,
        source_name: str = "frame",
    ) -> tuple[DetectedFaceEmbedding, ...]:
        """Return zero or more aligned face embeddings from one BGR frame."""
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise FacePhotoError(f"source must be a non-empty BGR image: {source_name}")
        height, width = image.shape[:2]
        max_input_dimension = getattr(self, "max_input_dimension", max(height, width))
        scale = min(1.0, max_input_dimension / max(height, width))
        if scale < 1:
            detection_image = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = image
        detection_height, detection_width = detection_image.shape[:2]
        try:
            self._detector.setInputSize((detection_width, detection_height))
            _, faces = self._detector.detect(detection_image)
        except cv2.error as error:
            raise FacePhotoError(f"face detection failed: {source_name}") from error

        if faces is None:
            return ()
        return tuple(
            self._extract_detected_face(
                detection_image,
                raw_face,
                source_name=source_name,
                bounds_scale=1 / scale,
                output_size=(width, height),
            )
            for raw_face in faces
        )

    def _extract_detected_face(
        self,
        image: NDArray[np.uint8],
        raw_face: NDArray[np.float32],
        *,
        source_name: str,
        bounds_scale: float,
        output_size: tuple[int, int],
    ) -> DetectedFaceEmbedding:
        face = np.asarray(raw_face, dtype=np.float32)
        if face.size < 15:
            raise FacePhotoError(f"face detector returned incomplete landmarks: {source_name}")
        confidence = float(face[14])
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise FacePhotoError(f"face detector returned invalid confidence: {source_name}")

        try:
            aligned = self._recognizer.alignCrop(image, face)
            feature = self._recognizer.feature(aligned)
        except cv2.error as error:
            raise FacePhotoError(f"face alignment or embedding failed: {source_name}") from error
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size != SFACE_EMBEDDING_DIMENSION:
            raise FacePhotoError(
                f"SFace must return {SFACE_EMBEDDING_DIMENSION} values; "
                f"received {vector.size}: {source_name}"
            )
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 0:
            raise FacePhotoError(f"SFace returned an invalid embedding: {source_name}")
        output_width, output_height = output_size
        x = max(0, min(round(float(face[0]) * bounds_scale), output_width))
        y = max(0, min(round(float(face[1]) * bounds_scale), output_height))
        right = max(
            x,
            min(round(float(face[0] + face[2]) * bounds_scale), output_width),
        )
        bottom = max(
            y,
            min(round(float(face[1] + face[3]) * bounds_scale), output_height),
        )
        return DetectedFaceEmbedding(
            bounds=FaceBounds(x=x, y=y, width=right - x, height=bottom - y),
            vector=tuple(float(value) for value in vector),
            detection_confidence=confidence,
        )


def _required_model_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FaceModelLoadError(f"{label} model file does not exist or is empty: {path}")
    return path


__all__ = [
    "SFACE_EMBEDDING_DIMENSION",
    "SFACE_MODEL_NAME",
    "SFACE_MODEL_VERSION",
    "YUNET_MODEL_NAME",
    "YUNET_MODEL_VERSION",
    "DetectedFaceEmbedding",
    "ExtractedFaceEmbedding",
    "FaceBounds",
    "FaceEmbeddingError",
    "FaceEmbeddingModelMetadata",
    "FaceModelLoadError",
    "FacePhotoError",
    "OpenCvSFaceExtractor",
]
