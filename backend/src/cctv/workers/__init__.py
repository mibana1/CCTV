"""Process orchestration and durable background-job boundaries.

Workers may coordinate analysis pipelines and own display-job retries, leases,
and recovery. Per-frame decoding, overlay, encoding, and restream logic remains
in :mod:`cctv.media`.
"""

from cctv.workers.local_video import (
    FrameConsumer,
    LocalVideoWorker,
    LocalVideoWorkerResult,
    WorkerStatus,
    WorkerStopReason,
)

__all__ = [
    "FrameConsumer",
    "LocalVideoWorker",
    "LocalVideoWorkerResult",
    "WorkerStatus",
    "WorkerStopReason",
]
