"""Process orchestration and durable background-job boundaries.

Workers may coordinate analysis pipelines and own display-job retries, leases,
and recovery. Per-frame decoding, overlay, encoding, and restream logic remains
in :mod:`cctv.media`.
"""

from cctv.workers.analysis_supervisor import (
    AnalysisSupervisor,
    AnalysisSupervisorSnapshot,
    AnalysisWorkerSnapshot,
    AnalysisWorkerStatus,
    camera_rtsp_url,
)
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
    "AnalysisSupervisor",
    "AnalysisSupervisorSnapshot",
    "AnalysisWorkerSnapshot",
    "AnalysisWorkerStatus",
    "FrameConsumer",
    "LocalVideoWorker",
    "LocalVideoWorkerResult",
    "RtspStopReason",
    "RtspWorker",
    "RtspWorkerResult",
    "SequentialFrameConsumer",
    "WorkerStatus",
    "WorkerStopReason",
    "camera_rtsp_url",
]
