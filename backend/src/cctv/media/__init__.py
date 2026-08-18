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

__all__ = [
    "DecodedFrame",
    "LocalVideoDecodeError",
    "LocalVideoDecoder",
    "LocalVideoError",
    "LocalVideoOpenError",
    "VideoMetadata",
    "iter_local_video_frames",
]
