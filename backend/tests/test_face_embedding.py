from pathlib import Path

import cv2
import numpy as np
import pytest

from cctv.identity import (
    SFACE_EMBEDDING_DIMENSION,
    FaceModelLoadError,
    FacePhotoError,
    OpenCvSFaceExtractor,
)


class _Detector:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, value: tuple[int, int]) -> None:
        self.input_size = value

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        del image
        return 1, self.faces


class _Recognizer:
    def __init__(self, feature: np.ndarray) -> None:
        self.output = feature
        self.aligned_face: np.ndarray | None = None

    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        del face
        self.aligned_face = image[:112, :112]
        return self.aligned_face

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        assert aligned is self.aligned_face
        return self.output


def _extractor(
    *,
    faces: np.ndarray | None,
    feature: np.ndarray | None = None,
) -> OpenCvSFaceExtractor:
    extractor = object.__new__(OpenCvSFaceExtractor)
    extractor._detector = _Detector(faces)
    extractor._recognizer = _Recognizer(
        feature
        if feature is not None
        else np.arange(1, SFACE_EMBEDDING_DIMENSION + 1, dtype=np.float32)[None, :]
    )
    return extractor


def _face(confidence: float = 0.97) -> np.ndarray:
    return np.array(
        [[10, 10, 100, 100, 30, 40, 70, 40, 50, 60, 35, 80, 65, 80, confidence]],
        dtype=np.float32,
    )


def test_sface_extractor_returns_one_valid_128_dimension_feature() -> None:
    extractor = _extractor(faces=_face())
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = extractor.extract(image, source_name="front.jpg")

    assert len(result.vector) == SFACE_EMBEDDING_DIMENSION
    assert result.vector[0] == pytest.approx(1)
    assert result.detection_confidence == pytest.approx(0.97)
    assert extractor._detector.input_size == (320, 240)


def test_sface_extractor_returns_all_video_frame_faces_with_bounds() -> None:
    faces = np.concatenate((_face(), _face(0.95)))
    faces[1, :4] = [-5, 20, 50, 60]
    extractor = _extractor(faces=faces)

    results = extractor.extract_many(
        np.zeros((100, 120, 3), dtype=np.uint8),
        source_name="sample-1",
    )

    assert len(results) == 2
    assert results[0].bounds == results[0].bounds.__class__(10, 10, 100, 90)
    assert results[1].bounds == results[1].bounds.__class__(0, 20, 45, 60)
    assert results[1].detection_confidence == pytest.approx(0.95)


def test_sface_extractor_returns_empty_video_frame_result() -> None:
    extractor = _extractor(faces=None)

    assert extractor.extract_many(np.zeros((100, 100, 3), dtype=np.uint8)) == ()


def test_sface_extractor_resizes_large_input_and_restores_original_bounds() -> None:
    extractor = _extractor(faces=_face())
    extractor.max_input_dimension = 320

    result = extractor.extract_many(np.zeros((400, 800, 3), dtype=np.uint8))[0]

    assert extractor._detector.input_size == (320, 160)
    assert (result.bounds.x, result.bounds.y, result.bounds.width, result.bounds.height) == (
        25,
        25,
        250,
        250,
    )


@pytest.mark.parametrize(
    ("faces", "expected_count"),
    [
        (None, 0),
        (np.concatenate((_face(), _face(0.95))), 2),
    ],
)
def test_sface_extractor_requires_exactly_one_face(
    faces: np.ndarray | None,
    expected_count: int,
) -> None:
    extractor = _extractor(faces=faces)

    with pytest.raises(FacePhotoError, match=f"found {expected_count}"):
        extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8), source_name="photo.jpg")


def test_sface_extractor_rejects_invalid_feature_shape() -> None:
    extractor = _extractor(faces=_face(), feature=np.ones((1, 64), dtype=np.float32))

    with pytest.raises(FacePhotoError, match="must return 128"):
        extractor.extract(np.zeros((120, 120, 3), dtype=np.uint8))


def test_sface_extractor_decodes_file_without_exposing_vector(tmp_path: Path) -> None:
    extractor = _extractor(faces=_face())
    ok, encoded = cv2.imencode(".jpg", np.zeros((120, 120, 3), dtype=np.uint8))
    assert ok is True
    photo_path = tmp_path / "인물사진.jpg"
    photo_path.write_bytes(encoded.tobytes())

    result = extractor.extract_file(photo_path)

    assert len(result.vector) == SFACE_EMBEDDING_DIMENSION


def test_sface_extractor_requires_both_model_files(tmp_path: Path) -> None:
    with pytest.raises(FaceModelLoadError, match="YuNet"):
        OpenCvSFaceExtractor(tmp_path / "missing-yunet.onnx", tmp_path / "missing-sface.onnx")
