"""Interface implemented by interchangeable object detectors."""

from typing import Protocol, runtime_checkable

from cctv.inference.models import DetectorMetadata, DetectorRunSummary, FrameDetections
from cctv.media import DecodedFrame


@runtime_checkable
class ObjectDetector(Protocol):
    """Minimal boundary consumed by local-video and RTSP workers."""

    @property
    def metadata(self) -> DetectorMetadata:
        """Return stable model metadata before the first frame is analyzed."""

    @property
    def summary(self) -> DetectorRunSummary:
        """Return current bounded aggregate statistics."""

    def analyze(self, frame: DecodedFrame) -> FrameDetections:
        """Analyze one frame and return detector-independent results."""


__all__ = ["ObjectDetector"]
