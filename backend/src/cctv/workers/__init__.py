"""Process orchestration and durable background-job boundaries.

Workers may coordinate analysis pipelines and own display-job retries, leases,
and recovery. Per-frame decoding, overlay, encoding, and restream logic remains
in :mod:`cctv.media`.
"""

from cctv.workers.local_video import (
    FrameConsumer,
    LocalVideoWorker,
    LocalVideoWorkerResult,
    SequentialFrameConsumer,
    WorkerStatus,
    WorkerStopReason,
)
from cctv.workers.rtsp import RtspStopReason, RtspWorker, RtspWorkerResult

__all__ = [
    "FrameConsumer",
    "LocalVideoWorker",
    "LocalVideoWorkerResult",
    "RtspStopReason",
    "RtspWorker",
    "RtspWorkerResult",
    "SequentialFrameConsumer",
    "WorkerStatus",
    "WorkerStopReason",
]
