"""Atomic JPEG snapshot storage for sampled video frames."""

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2

from cctv.media.decode import DecodedFrame

logger = logging.getLogger(__name__)

_UNSAFE_RUN_NAME = re.compile(r"[^\w.-]+", flags=re.UNICODE)


class SnapshotError(RuntimeError):
    """Base error raised while encoding or writing a sampled frame."""


class SnapshotEncodingError(SnapshotError):
    """Raised when OpenCV cannot encode a frame as JPEG."""


class SnapshotWriteError(SnapshotError):
    """Raised when a snapshot cannot be written without overwriting data."""


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """Metadata for one JPEG snapshot successfully persisted to disk."""

    path: Path
    source_index: int
    sample_index: int
    timestamp_seconds: float
    size_bytes: int


class SnapshotWriter:
    """Encode sampled frames as JPEG files in one new run directory.

    The output directory must not already exist. Each JPEG is fully encoded in
    memory, written to a unique temporary file, and atomically moved into place.
    Existing snapshots are never overwritten.
    """

    def __init__(self, output_dir: str | Path, *, jpeg_quality: int = 90) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.jpeg_quality = jpeg_quality
        self.saved_count = 0
        self.first_path: Path | None = None
        self.last_path: Path | None = None
        self._initialized = False

    def __call__(self, frame: DecodedFrame) -> None:
        """Allow the writer to be used directly as a worker frame consumer."""
        self.save(frame)

    def save(self, frame: DecodedFrame) -> SnapshotRecord:
        """Encode and atomically persist one sampled frame."""
        if frame.image.size == 0:
            raise SnapshotEncodingError("cannot encode an empty frame")

        encode_ok, encoded = cv2.imencode(
            ".jpg",
            frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encode_ok:
            raise SnapshotEncodingError(
                f"JPEG encoding failed for sample index {frame.sample_index}"
            )

        self._initialize_output_directory()
        output_path = self.output_dir / self.filename_for(frame)
        if output_path.exists():
            raise SnapshotWriteError(f"snapshot already exists: {output_path.name}")

        temporary_path = self.output_dir / f".{output_path.name}.{uuid4().hex}.tmp"
        try:
            with temporary_path.open("xb") as temporary_file:
                temporary_file.write(encoded.tobytes())
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, output_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            if isinstance(error, SnapshotError):
                raise
            raise SnapshotWriteError(
                f"snapshot write failed for sample index {frame.sample_index}"
            ) from error

        record = SnapshotRecord(
            path=output_path,
            source_index=frame.source_index,
            sample_index=frame.sample_index,
            timestamp_seconds=frame.timestamp_seconds,
            size_bytes=output_path.stat().st_size,
        )
        self.saved_count += 1
        if self.first_path is None:
            self.first_path = output_path
        self.last_path = output_path
        logger.debug(
            "Sample frame snapshot saved",
            extra={
                "event": "snapshot_saved",
                "snapshot_path": output_path,
                "source_index": frame.source_index,
                "sample_index": frame.sample_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "size_bytes": record.size_bytes,
            },
        )
        return record

    @staticmethod
    def filename_for(frame: DecodedFrame) -> str:
        """Build a sortable filename from sample, source frame, and timestamp."""
        timestamp_ms = max(0, round(frame.timestamp_seconds * 1_000))
        return (
            f"sample_{frame.sample_index:06d}_"
            f"frame_{frame.source_index:09d}_"
            f"t{timestamp_ms:012d}ms.jpg"
        )

    def _initialize_output_directory(self) -> None:
        if self._initialized:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise SnapshotWriteError(
                f"snapshot output directory already exists: {self.output_dir}"
            ) from error
        self._initialized = True
        logger.info(
            "Snapshot output initialized",
            extra={
                "event": "snapshot_output_initialized",
                "snapshot_dir": self.output_dir,
                "jpeg_quality": self.jpeg_quality,
            },
        )


def build_snapshot_run_directory(
    snapshot_root: str | Path,
    video_path: str | Path,
) -> Path:
    """Return a unique, filesystem-safe directory for one worker run."""
    source_name = _UNSAFE_RUN_NAME.sub("_", Path(video_path).stem).strip("._-") or "video"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        Path(snapshot_root).expanduser().resolve() / f"{source_name}_{timestamp}_{uuid4().hex[:8]}"
    )
