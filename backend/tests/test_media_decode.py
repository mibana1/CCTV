from pathlib import Path

import cv2
import numpy as np
import pytest

from cctv.media import (
    LocalVideoDecodeError,
    LocalVideoDecoder,
    LocalVideoOpenError,
    iter_local_video_frames,
)


class FailingCapture:
    """Minimal capture that reports three frames but stops after the first."""

    def __init__(self) -> None:
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def get(self, property_id: int) -> float:
        properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 32.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            cv2.CAP_PROP_FPS: 4.0,
            cv2.CAP_PROP_FRAME_COUNT: 3.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return max(self.read_count - 1, 0) * 250.0
        return properties.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_count > 0:
            return False, None
        self.read_count += 1
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def create_test_video(path: Path, *, fps: float = 4.0, frame_count: int = 12) -> Path:
    width, height = 32, 24
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.fail("OpenCV MJPG test writer could not be opened")

    try:
        for index in range(frame_count):
            image = np.full((height, width, 3), index * 10, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()

    return path


def test_local_video_decoder_reads_metadata_and_samples_by_timestamp(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")
    decoder = LocalVideoDecoder(video_path, sample_fps=2)

    with decoder:
        metadata = decoder.metadata
        frames = list(decoder.frames())

    assert decoder.is_open is False
    assert metadata.path == video_path.resolve()
    assert metadata.codec == "mjpg"
    assert metadata.width == 32
    assert metadata.height == 24
    assert metadata.source_fps == pytest.approx(4)
    assert metadata.frame_count == 12
    assert metadata.duration_seconds == pytest.approx(3)
    assert [frame.source_index for frame in frames] == [0, 2, 4, 6, 8, 10]
    assert [frame.sample_index for frame in frames] == list(range(6))
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0, 0.5, 1, 1.5, 2, 2.5])
    assert all(frame.image.shape == (24, 32, 3) for frame in frames)


def test_local_video_decoder_does_not_duplicate_frames(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")

    frames = list(iter_local_video_frames(video_path, sample_fps=8))

    assert len(frames) == 12
    assert [frame.source_index for frame in frames] == list(range(12))


def test_local_video_decoder_rejects_missing_file(tmp_path: Path) -> None:
    decoder = LocalVideoDecoder(tmp_path / "missing.mp4", sample_fps=2)

    with pytest.raises(LocalVideoOpenError, match="does not exist"):
        decoder.open()

    assert decoder.is_open is False


@pytest.mark.parametrize("sample_fps", [0, -1, float("inf"), float("nan")])
def test_local_video_decoder_rejects_invalid_sample_fps(sample_fps: float) -> None:
    with pytest.raises(ValueError, match="sample_fps"):
        LocalVideoDecoder("sample.mp4", sample_fps=sample_fps)


def test_local_video_decoder_requires_open_context(tmp_path: Path) -> None:
    decoder = LocalVideoDecoder(tmp_path / "sample.avi", sample_fps=2)

    with pytest.raises(RuntimeError, match="not open"):
        list(decoder.frames())


def test_local_video_decoder_reports_premature_decode_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "truncated.avi"
    video_path.write_bytes(b"placeholder")
    capture = FailingCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    decoder = LocalVideoDecoder(video_path, sample_fps=2)
    with pytest.raises(LocalVideoDecodeError, match="decoded 1 of 3"), decoder:
        list(decoder.frames())

    assert capture.released is True
    assert decoder.is_open is False
