"""Local video worker orchestration and command-line entry point."""

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from time import perf_counter

from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import get_settings
from cctv.media import DecodedFrame, LocalVideoDecoder, VideoMetadata

logger = logging.getLogger(__name__)

FrameConsumer = Callable[[DecodedFrame], None]


class WorkerStatus(StrEnum):
    """Lifecycle state for a single-use local video worker."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class WorkerStopReason(StrEnum):
    """Reason a local video worker stopped processing frames."""

    END_OF_STREAM = "end_of_stream"
    SAMPLE_LIMIT = "sample_limit"
    STOP_REQUESTED = "stop_requested"


@dataclass(frozen=True, slots=True)
class LocalVideoWorkerResult:
    """Summary returned after a worker finishes without raising an error."""

    status: WorkerStatus
    stop_reason: WorkerStopReason
    video_path: Path
    sample_fps: float
    processed_samples: int
    first_source_index: int | None
    last_source_index: int | None
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    elapsed_seconds: float
    metadata: VideoMetadata | None


def discard_frame(frame: DecodedFrame) -> None:
    """Default consumer used for decode-only worker runs."""
    del frame


class LocalVideoWorker:
    """Run one local video through the decoder outside the API request path.

    The worker is deliberately single-use. A frame consumer can be supplied by
    later snapshot or inference stages without moving decoding into the worker.
    Stop requests are checked between sampled frames and do not interrupt a
    native OpenCV read already in progress.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        sample_fps: float,
        frame_consumer: FrameConsumer | None = None,
        max_samples: int | None = None,
    ) -> None:
        if not isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be a finite number greater than zero")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be greater than zero when provided")

        self.video_path = Path(video_path).expanduser().resolve()
        self.sample_fps = float(sample_fps)
        self.frame_consumer = frame_consumer or discard_frame
        self.max_samples = max_samples
        self._status = WorkerStatus.IDLE
        self._status_lock = Lock()
        self._stop_event = Event()

    @property
    def status(self) -> WorkerStatus:
        """Return the current lifecycle state in a thread-safe manner."""
        with self._status_lock:
            return self._status

    def stop(self) -> None:
        """Request cooperative shutdown between sampled frames."""
        self._stop_event.set()
        logger.info(
            "Local video worker stop requested",
            extra={
                "event": "local_video_worker_stop_requested",
                "video_path": self.video_path,
            },
        )

    def execute(self) -> LocalVideoWorkerResult:
        """Decode and dispatch sampled frames until completion or a stop condition."""
        self._begin_execution()
        started_at = perf_counter()
        metadata: VideoMetadata | None = None
        processed_samples = 0
        first_source_index: int | None = None
        last_source_index: int | None = None
        first_timestamp_seconds: float | None = None
        last_timestamp_seconds: float | None = None
        stop_reason = WorkerStopReason.STOP_REQUESTED

        logger.info(
            "Local video worker started",
            extra={
                "event": "local_video_worker_started",
                "video_path": self.video_path,
                "sample_fps": self.sample_fps,
                "max_samples": self.max_samples,
            },
        )

        try:
            if not self._stop_event.is_set():
                with LocalVideoDecoder(self.video_path, sample_fps=self.sample_fps) as decoder:
                    metadata = decoder.metadata
                    for frame in decoder.frames():
                        if self._stop_event.is_set():
                            stop_reason = WorkerStopReason.STOP_REQUESTED
                            break

                        self.frame_consumer(frame)
                        processed_samples += 1
                        if first_source_index is None:
                            first_source_index = frame.source_index
                            first_timestamp_seconds = frame.timestamp_seconds
                        last_source_index = frame.source_index
                        last_timestamp_seconds = frame.timestamp_seconds

                        if self._stop_event.is_set():
                            stop_reason = WorkerStopReason.STOP_REQUESTED
                            break
                        if self.max_samples is not None and processed_samples >= self.max_samples:
                            stop_reason = WorkerStopReason.SAMPLE_LIMIT
                            break
                    else:
                        stop_reason = WorkerStopReason.END_OF_STREAM

            final_status = (
                WorkerStatus.STOPPED
                if stop_reason is WorkerStopReason.STOP_REQUESTED
                else WorkerStatus.COMPLETED
            )
            self._set_status(final_status)
        except Exception:
            self._set_status(WorkerStatus.FAILED)
            logger.exception(
                "Local video worker failed",
                extra={
                    "event": "local_video_worker_failed",
                    "video_path": self.video_path,
                    "sample_fps": self.sample_fps,
                    "processed_samples": processed_samples,
                },
            )
            raise

        result = LocalVideoWorkerResult(
            status=final_status,
            stop_reason=stop_reason,
            video_path=self.video_path,
            sample_fps=self.sample_fps,
            processed_samples=processed_samples,
            first_source_index=first_source_index,
            last_source_index=last_source_index,
            first_timestamp_seconds=first_timestamp_seconds,
            last_timestamp_seconds=last_timestamp_seconds,
            elapsed_seconds=round(perf_counter() - started_at, 6),
            metadata=metadata,
        )
        logger.info(
            "Local video worker finished",
            extra={
                "event": (
                    "local_video_worker_stopped"
                    if final_status is WorkerStatus.STOPPED
                    else "local_video_worker_completed"
                ),
                "video_path": self.video_path,
                "status": final_status,
                "stop_reason": stop_reason,
                "processed_samples": processed_samples,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
        return result

    def _begin_execution(self) -> None:
        with self._status_lock:
            if self._status is not WorkerStatus.IDLE:
                raise RuntimeError("local video worker instances can only be executed once")
            self._status = WorkerStatus.RUNNING

    def _set_status(self, status: WorkerStatus) -> None:
        with self._status_lock:
            self._status = status


def positive_integer(value: str) -> int:
    """Parse a positive integer for an argparse option."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for a local video worker run."""
    parser = argparse.ArgumentParser(description="Decode and sample one local video file")
    parser.add_argument("video_path", nargs="?", type=Path)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--max-samples", type=positive_integer)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    """Run a decode-only local video worker from the command line."""
    settings = get_settings()
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    video_path = arguments.video_path or settings.local_video_path
    if video_path is None:
        parser.error("video_path or CCTV_LOCAL_VIDEO_PATH is required")

    configure_logging(
        level=settings.log_level,
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    try:
        result = LocalVideoWorker(
            video_path,
            sample_fps=(
                arguments.sample_fps if arguments.sample_fps is not None else settings.analysis_fps
            ),
            max_samples=arguments.max_samples,
        ).execute()
        print(json.dumps(asdict(result), default=str, ensure_ascii=False, sort_keys=True))
    finally:
        shutdown_logging()
