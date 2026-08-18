from collections.abc import Iterator

import cv2
import numpy as np
import pytest

from cctv.media import RtspConnectionError, RtspStreamReader


class FakeCapture:
    def __init__(self, frame_count: int, *, opened: bool = True) -> None:
        self.frames = [np.full((4, 6, 3), index, dtype=np.uint8) for index in range(frame_count)]
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class ImmediateEvent:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float) -> bool:
        self.delays.append(timeout)
        return False


def sequence_clock(values: list[float]) -> Iterator[float]:
    yield from values
    while True:
        yield values[-1]


def test_rtsp_reader_reconnects_without_resetting_frame_indexes() -> None:
    captures = [FakeCapture(2), FakeCapture(2)]
    calls: list[tuple[str, int, list[int]]] = []

    def capture_factory(url: str, backend: int, parameters: list[int]) -> FakeCapture:
        calls.append((url, backend, parameters))
        return captures[len(calls) - 1]

    clock_values = sequence_clock([100.0, 100.0, 100.1, 100.5])
    reader = RtspStreamReader(
        "rtsp://user:password@camera.local/stream",
        source_name="camera-1",
        open_timeout_seconds=3,
        read_timeout_seconds=4,
        reconnect_initial_seconds=1,
        reconnect_max_seconds=2,
        reconnect_jitter_ratio=0,
        capture_factory=capture_factory,
        clock=lambda: next(clock_values),
    )
    stop_event = ImmediateEvent()
    frame_iterator = reader.frames(stop_event=stop_event)  # type: ignore[arg-type]
    frames = [next(frame_iterator) for _ in range(3)]
    frame_iterator.close()

    assert [frame.source_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0, 0.1, 0.5])
    assert reader.statistics.connection_count == 2
    assert reader.statistics.reconnect_count == 1
    assert reader.statistics.decoded_frames == 3
    assert stop_event.delays == [1]
    assert all(capture.released for capture in captures)
    assert calls[0][1] == cv2.CAP_FFMPEG
    assert calls[0][2] == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        3_000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        4_000,
    ]


def test_rtsp_reader_exhausts_retry_budget_without_logging_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failed_capture = FakeCapture(0, opened=False)
    reader = RtspStreamReader(
        "rtsp://secret-user:secret-password@camera.local/stream?token=secret-token",
        source_name="camera-1",
        max_retries=0,
        capture_factory=lambda url, backend, parameters: failed_capture,
    )

    with (
        caplog.at_level("INFO", logger="cctv.media.ingest"),
        pytest.raises(
            RtspConnectionError,
            match="camera-1",
        ),
    ):
        next(reader.frames())

    log_text = caplog.text
    assert "secret-user" not in log_text
    assert "secret-password" not in log_text
    assert "secret-token" not in log_text
    assert failed_capture.released is True


@pytest.mark.parametrize("url", ["http://camera/stream", "rtsp:///stream", "not-a-url"])
def test_rtsp_reader_rejects_invalid_urls(url: str) -> None:
    with pytest.raises(ValueError, match="rtsp"):
        RtspStreamReader(url, source_name="camera")
