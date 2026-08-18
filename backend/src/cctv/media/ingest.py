"""RTSP connection, timeout, reconnect, and source-health boundary."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import isfinite
from threading import Event
from time import monotonic
from typing import Protocol
from urllib.parse import urlsplit

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RtspIngestError(RuntimeError):
    """Base error raised by RTSP ingestion."""


class RtspConnectionError(RtspIngestError):
    """Raised after the configured consecutive reconnect budget is exhausted."""


class _Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str, int, list[int]], _Capture]
Clock = Callable[[], float]
Jitter = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class RawRtspFrame:
    """One decoded RTSP frame before analysis-FPS sampling."""

    source_index: int
    timestamp_seconds: float
    image: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class RtspIngestStatistics:
    """Current bounded counters for one reader instance."""

    connection_count: int
    reconnect_count: int
    decoded_frames: int


class RtspStreamReader:
    """Read RTSP frames with OpenCV timeouts and bounded exponential reconnects.

    The reader is synchronous and owns no Python-side prefetch queue. When the
    downstream analyzer is slower than the stream, memory therefore remains
    bounded instead of accumulating decoded frames.
    """

    def __init__(
        self,
        url: str,
        *,
        source_name: str,
        open_timeout_seconds: float = 10,
        read_timeout_seconds: float = 10,
        reconnect_initial_seconds: float = 1,
        reconnect_max_seconds: float = 30,
        reconnect_jitter_ratio: float = 0.2,
        max_retries: int = 20,
        capture_factory: CaptureFactory | None = None,
        clock: Clock = monotonic,
        jitter: Jitter = random.uniform,
    ) -> None:
        parsed = urlsplit(url)
        if parsed.scheme.casefold() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("url must use rtsp:// or rtsps:// and include a host")
        if not source_name.strip():
            raise ValueError("source_name must not be empty")
        for name, value in (
            ("open_timeout_seconds", open_timeout_seconds),
            ("read_timeout_seconds", read_timeout_seconds),
            ("reconnect_initial_seconds", reconnect_initial_seconds),
            ("reconnect_max_seconds", reconnect_max_seconds),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite number greater than zero")
        if reconnect_max_seconds < reconnect_initial_seconds:
            raise ValueError(
                "reconnect_max_seconds must be greater than or equal to reconnect_initial_seconds"
            )
        if not isfinite(reconnect_jitter_ratio) or not 0 <= reconnect_jitter_ratio <= 1:
            raise ValueError("reconnect_jitter_ratio must be between zero and one")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        self._url = url
        self.source_name = source_name.strip()
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.read_timeout_seconds = float(read_timeout_seconds)
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.reconnect_jitter_ratio = float(reconnect_jitter_ratio)
        self.max_retries = max_retries
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock = clock
        self._jitter = jitter
        self._started = False
        self._connection_count = 0
        self._reconnect_count = 0
        self._decoded_frames = 0

    @property
    def statistics(self) -> RtspIngestStatistics:
        """Return counters without exposing the source URL or credentials."""
        return RtspIngestStatistics(
            connection_count=self._connection_count,
            reconnect_count=self._reconnect_count,
            decoded_frames=self._decoded_frames,
        )

    def frames(self, *, stop_event: Event | None = None) -> Iterator[RawRtspFrame]:
        """Yield decoded frames, reconnecting until stopped or the budget is exhausted."""
        if self._started:
            raise RuntimeError("RTSP reader instances can only be consumed once")
        self._started = True
        requested_stop = stop_event or Event()
        started_at = self._clock()
        consecutive_failures = 0
        retry_delay = self.reconnect_initial_seconds

        while not requested_stop.is_set():
            capture = self._open_capture()
            if capture is None:
                consecutive_failures += 1
                self._raise_if_retry_budget_exhausted(consecutive_failures)
                self._reconnect_count += 1
                if self._wait_for_retry(requested_stop, retry_delay, consecutive_failures):
                    break
                retry_delay = min(retry_delay * 2, self.reconnect_max_seconds)
                continue

            self._connection_count += 1
            frames_on_connection = 0
            logger.info(
                "RTSP source connected",
                extra={
                    "event": "rtsp_source_connected",
                    "source_name": self.source_name,
                    "connection_count": self._connection_count,
                },
            )
            try:
                while not requested_stop.is_set():
                    try:
                        read_ok, image = capture.read()
                    except cv2.error:
                        read_ok, image = False, None
                    if not read_ok or image is None or image.size == 0:
                        break

                    frame = RawRtspFrame(
                        source_index=self._decoded_frames,
                        timestamp_seconds=max(0.0, self._clock() - started_at),
                        image=image,
                    )
                    self._decoded_frames += 1
                    frames_on_connection += 1
                    yield frame
            finally:
                capture.release()

            if requested_stop.is_set():
                break

            consecutive_failures = 1 if frames_on_connection else consecutive_failures + 1
            self._raise_if_retry_budget_exhausted(consecutive_failures)
            retry_delay = self.reconnect_initial_seconds if frames_on_connection else retry_delay
            self._reconnect_count += 1
            logger.warning(
                "RTSP source disconnected",
                extra={
                    "event": "rtsp_source_disconnected",
                    "source_name": self.source_name,
                    "frames_on_connection": frames_on_connection,
                    "reconnect_count": self._reconnect_count,
                },
            )
            if self._wait_for_retry(requested_stop, retry_delay, consecutive_failures):
                break
            retry_delay = min(retry_delay * 2, self.reconnect_max_seconds)

    def _open_capture(self) -> _Capture | None:
        timeout_parameters = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            round(self.open_timeout_seconds * 1_000),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            round(self.read_timeout_seconds * 1_000),
        ]
        try:
            capture = self._capture_factory(self._url, cv2.CAP_FFMPEG, timeout_parameters)
        except cv2.error:
            capture = None
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            logger.warning(
                "RTSP source connection failed",
                extra={
                    "event": "rtsp_source_connection_failed",
                    "source_name": self.source_name,
                },
            )
            return None
        return capture

    def _raise_if_retry_budget_exhausted(self, consecutive_failures: int) -> None:
        if consecutive_failures <= self.max_retries:
            return
        logger.error(
            "RTSP reconnect budget exhausted",
            extra={
                "event": "rtsp_reconnect_exhausted",
                "source_name": self.source_name,
                "max_retries": self.max_retries,
            },
        )
        raise RtspConnectionError(
            f"RTSP source '{self.source_name}' exceeded {self.max_retries} reconnect attempts"
        )

    def _wait_for_retry(
        self,
        stop_event: Event,
        base_delay: float,
        consecutive_failures: int,
    ) -> bool:
        jitter = base_delay * self.reconnect_jitter_ratio
        delay = max(0.0, self._jitter(base_delay - jitter, base_delay + jitter))
        logger.info(
            "RTSP reconnect scheduled",
            extra={
                "event": "rtsp_reconnect_scheduled",
                "source_name": self.source_name,
                "retry_in_seconds": round(delay, 3),
                "consecutive_failures": consecutive_failures,
            },
        )
        return stop_event.wait(delay)
