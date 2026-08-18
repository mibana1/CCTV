"""Long-running CCTV media pipeline boundaries.

Frame ingestion, decoding, overlay rendering, encoding, and RTSP publication
belong in this package. Durable job retries and Hiperwall layout operations do
not.

See ``docs/architecture/MEDIA_PIPELINE.md`` before adding implementations.
"""

from cctv.media.decode import (
    DecodedFrame,
    LocalVideoDecodeError,
    LocalVideoDecoder,
    LocalVideoError,
    LocalVideoOpenError,
    VideoMetadata,
    iter_local_video_frames,
)
from cctv.media.snapshot import (
    SnapshotEncodingError,
    SnapshotError,
    SnapshotRecord,
    SnapshotWriteError,
    SnapshotWriter,
    build_snapshot_run_directory,
)

__all__ = [
    "DecodedFrame",
    "LocalVideoDecodeError",
    "LocalVideoDecoder",
    "LocalVideoError",
    "LocalVideoOpenError",
    "SnapshotEncodingError",
    "SnapshotError",
    "SnapshotRecord",
    "SnapshotWriteError",
    "SnapshotWriter",
    "VideoMetadata",
    "build_snapshot_run_directory",
    "iter_local_video_frames",
]
