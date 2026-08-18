"""Decode, timestamp, bounded-buffer, and frame-sampling boundary.

The implementation must drop stale frames rather than allow unbounded latency
growth when downstream analysis cannot keep up.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_TIMESTAMP_EPSILON_SECONDS = 1e-9


class LocalVideoError(RuntimeError):
    """Base error raised while opening or decoding a local video."""


class LocalVideoOpenError(LocalVideoError):
    """Raised when a local video cannot be opened or has invalid metadata."""


class LocalVideoDecodeError(LocalVideoError):
    """Raised when decoding stops before the reported end of a video."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Validated metadata read from a local video container."""

    path: Path
    codec: str
    width: int
    height: int
    source_fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One sampled BGR frame and its position in the source video."""

    source_index: int
    sample_index: int
    timestamp_seconds: float
    image: NDArray[np.uint8]


class LocalVideoDecoder:
    """Pull-based local video decoder with timestamp-aware FPS sampling.

    Frames are decoded synchronously and no prefetch queue is created. This
    keeps decoder memory bounded to the current frame while allowing a future
    worker to decide how sampled frames are buffered for inference.
    """

    def __init__(self, path: str | Path, *, sample_fps: float) -> None:
        if not isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be a finite number greater than zero")

        self.path = Path(path).expanduser().resolve()
        self.sample_fps = float(sample_fps)
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None
        self._started = False

    @property
    def is_open(self) -> bool:
        """Return whether the underlying OpenCV capture is currently open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def metadata(self) -> VideoMetadata:
        """Return source metadata after the decoder has been opened."""
        if self._metadata is None:
            raise RuntimeError("local video decoder is not open")
        return self._metadata

    def open(self) -> Self:
        """Open the local file and validate the metadata required for sampling."""
        if self.is_open:
            return self

        if not self.path.is_file():
            logger.warning(
                "Local video open failed",
                extra={
                    "event": "local_video_open_failed",
                    "reason": "file_not_found",
                    "video_path": self.path,
                },
            )
            raise LocalVideoOpenError(f"local video file does not exist: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            logger.warning(
                "Local video open failed",
                extra={
                    "event": "local_video_open_failed",
                    "reason": "capture_not_opened",
                    "video_path": self.path,
                },
            )
            raise LocalVideoOpenError(f"local video could not be opened: {self.path}")

        try:
            metadata = self._read_metadata(capture)
        except Exception:
            capture.release()
            raise

        self._capture = capture
        self._metadata = metadata
        self._started = False
        logger.info(
            "Local video opened",
            extra={
                "event": "local_video_opened",
                "video_path": metadata.path,
                "codec": metadata.codec,
                "width": metadata.width,
                "height": metadata.height,
                "source_fps": metadata.source_fps,
                "sample_fps": self.sample_fps,
                "frame_count": metadata.frame_count,
                "duration_seconds": metadata.duration_seconds,
            },
        )
        return self

    def close(self) -> None:
        """Release the underlying native decoder resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self) -> Iterator[DecodedFrame]:
        """Yield sampled frames in source timestamp order without duplicating frames."""
        if not self.is_open or self._capture is None:
            raise RuntimeError("local video decoder is not open")
        if self._started:
            raise RuntimeError("local video frames can only be consumed once per open")

        self._started = True
        capture = self._capture
        metadata = self.metadata
        sample_interval_seconds = 1.0 / self.sample_fps
        next_sample_at = 0.0
        last_timestamp = -1.0
        decoded_count = 0
        sampled_count = 0

        while True:
            read_ok, image = capture.read()
            if not read_ok:
                if metadata.frame_count > 0 and decoded_count < metadata.frame_count:
                    logger.error(
                        "Local video decode failed",
                        extra={
                            "event": "local_video_decode_failed",
                            "video_path": metadata.path,
                            "decoded_frames": decoded_count,
                            "expected_frames": metadata.frame_count,
                        },
                    )
                    raise LocalVideoDecodeError(
                        "local video decode stopped before the reported final frame: "
                        f"decoded {decoded_count} of {metadata.frame_count}"
                    )
                break

            source_index = decoded_count
            timestamp_seconds = self._frame_timestamp(
                capture,
                source_index=source_index,
                source_fps=metadata.source_fps,
                last_timestamp=last_timestamp,
            )
            last_timestamp = timestamp_seconds
            decoded_count += 1

            if timestamp_seconds + _TIMESTAMP_EPSILON_SECONDS < next_sample_at:
                continue

            sampled_frame = DecodedFrame(
                source_index=source_index,
                sample_index=sampled_count,
                timestamp_seconds=timestamp_seconds,
                image=image,
            )
            sampled_count += 1
            while next_sample_at <= timestamp_seconds + _TIMESTAMP_EPSILON_SECONDS:
                next_sample_at += sample_interval_seconds
            yield sampled_frame

        logger.info(
            "Local video decode complete",
            extra={
                "event": "local_video_decode_completed",
                "video_path": metadata.path,
                "decoded_frames": decoded_count,
                "sampled_frames": sampled_count,
                "sample_fps": self.sample_fps,
            },
        )

    def __enter__(self) -> Self:
        """Open the decoder for context-managed use."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always release decoder resources when leaving a context."""
        self.close()

    def _read_metadata(self, capture: cv2.VideoCapture) -> VideoMetadata:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

        if width <= 0 or height <= 0:
            raise LocalVideoOpenError("local video has invalid frame dimensions")
        if not isfinite(source_fps) or source_fps <= 0:
            raise LocalVideoOpenError("local video has an invalid source FPS")

        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        codec = codec.strip("\x00 ").lower() or "unknown"
        duration_seconds = frame_count / source_fps if frame_count else 0.0

        return VideoMetadata(
            path=self.path,
            codec=codec,
            width=width,
            height=height,
            source_fps=source_fps,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )

    @staticmethod
    def _frame_timestamp(
        capture: cv2.VideoCapture,
        *,
        source_index: int,
        source_fps: float,
        last_timestamp: float,
    ) -> float:
        timestamp_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1_000
        fallback_timestamp = source_index / source_fps
        if (
            not isfinite(timestamp_seconds)
            or timestamp_seconds < 0
            or (source_index > 0 and timestamp_seconds <= last_timestamp)
        ):
            return fallback_timestamp
        return timestamp_seconds


def iter_local_video_frames(
    path: str | Path,
    *,
    sample_fps: float,
) -> Iterator[DecodedFrame]:
    """Open a local video and yield sampled frames with automatic cleanup."""
    with LocalVideoDecoder(path, sample_fps=sample_fps) as decoder:
        yield from decoder.frames()
