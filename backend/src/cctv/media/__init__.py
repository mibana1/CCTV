"""Long-running CCTV media pipeline boundaries.

Frame ingestion, decoding, overlay rendering, encoding, and RTSP publication
belong in this package. Durable job retries and Hiperwall layout operations do
not.

See ``docs/architecture/MEDIA_PIPELINE.md`` before adding implementations.
"""

from cctv.media.annotation import SnapshotOverlay, render_snapshot_annotations
from cctv.media.decode import (
    DecodedFrame,
    LocalVideoDecodeError,
    LocalVideoDecoder,
    LocalVideoError,
    LocalVideoOpenError,
    VideoMetadata,
    iter_local_video_frames,
)
from cctv.media.ingest import (
    RawRtspFrame,
    RtspConnectionError,
    RtspIngestError,
    RtspIngestStatistics,
    RtspStreamReader,
)
from cctv.media.snapshot import (
    SnapshotEncodingError,
    SnapshotError,
    SnapshotRecord,
    SnapshotWriteError,
    SnapshotWriter,
    build_snapshot_run_directory,
    sample_index_from_snapshot_name,
)

__all__ = [
    "DecodedFrame",
    "LocalVideoDecodeError",
    "LocalVideoDecoder",
    "LocalVideoError",
    "LocalVideoOpenError",
    "RawRtspFrame",
    "RtspConnectionError",
    "RtspIngestError",
    "RtspIngestStatistics",
    "RtspStreamReader",
    "SnapshotEncodingError",
    "SnapshotError",
    "SnapshotOverlay",
    "SnapshotRecord",
    "SnapshotWriteError",
    "SnapshotWriter",
    "VideoMetadata",
    "build_snapshot_run_directory",
    "iter_local_video_frames",
    "render_snapshot_annotations",
    "sample_index_from_snapshot_name",
]
