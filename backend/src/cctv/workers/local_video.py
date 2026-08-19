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
from cctv.core.settings import AppMode, get_settings
from cctv.db import (
    AnalysisRunRecord,
    AnalysisRunStatus,
    DetectionRepository,
    RuleRepository,
    initialize_database,
)
from cctv.hiperwall import HiperwallDryRunPlanner
from cctv.identity import (
    FaceMatchingConsumer,
    create_resolving_face_observation_sink,
    create_sface_matching_consumer,
)
from cctv.inference import (
    DetectorConfig,
    IoUTracker,
    ObjectDetector,
    create_detector,
    filter_detections_by_class,
)
from cctv.media import (
    DecodedFrame,
    LocalVideoDecoder,
    SnapshotWriter,
    VideoMetadata,
    build_snapshot_run_directory,
)
from cctv.rules import RuleEngine

logger = logging.getLogger(__name__)

FrameConsumer = Callable[[DecodedFrame], None]


class SequentialFrameConsumer:
    """Dispatch each frame to multiple synchronous, bounded consumers."""

    def __init__(self, *consumers: FrameConsumer) -> None:
        if not consumers:
            raise ValueError("at least one frame consumer is required")
        self.consumers = consumers

    def __call__(self, frame: DecodedFrame) -> None:
        for consumer in self.consumers:
            consumer(frame)


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


def jpeg_quality(value: str) -> int:
    """Parse a valid JPEG quality for an argparse option."""
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("value must be between 1 and 100")
    return parsed


def confidence_threshold(value: str) -> float:
    """Parse a confidence threshold in the half-open interval (0, 1]."""
    parsed = float(value)
    if not isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be greater than 0 and at most 1")
    return parsed


def nms_threshold(value: str) -> float:
    """Parse a non-maximum suppression threshold in the interval [0, 1]."""
    parsed = float(value)
    if not isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def model_input_size(value: str) -> int:
    """Parse the square ONNX model input size."""
    parsed = int(value)
    if not 32 <= parsed <= 4096:
        raise argparse.ArgumentTypeError("value must be between 32 and 4096")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for a local video worker run."""
    parser = argparse.ArgumentParser(description="Decode and sample one local video file")
    parser.add_argument("video_path", nargs="?", type=Path)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--max-samples", type=positive_integer)
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--jpeg-quality", type=jpeg_quality)
    parser.add_argument(
        "--analyze",
        "--analyze-yolo",
        dest="analyze",
        action="store_true",
        help="enable the configured object detector (--analyze-yolo is deprecated)",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--class-names-path", type=Path)
    parser.add_argument("--model-input-size", type=model_input_size)
    parser.add_argument("--confidence-threshold", type=confidence_threshold)
    parser.add_argument("--nms-threshold", type=nms_threshold)
    parser.add_argument(
        "--match-faces",
        action="store_true",
        help="detect faces and compare them with enabled registered identity embeddings",
    )
    parser.add_argument("--face-match-threshold", type=confidence_threshold)
    parser.add_argument("--face-match-margin", type=nms_threshold)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    """Run a local video worker with optional snapshots and object detection."""
    settings = get_settings()
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    video_path = arguments.video_path or settings.local_video_path
    if video_path is None:
        parser.error("video_path or CCTV_LOCAL_VIDEO_PATH is required")
    save_snapshots = arguments.save_snapshots or arguments.snapshot_dir is not None
    if arguments.jpeg_quality is not None and not save_snapshots:
        parser.error("--jpeg-quality requires --save-snapshots or --snapshot-dir")
    analyze_objects = arguments.analyze or arguments.model_path is not None
    analyze_objects = analyze_objects or settings.object_detection_enabled
    detector_options = (
        arguments.class_names_path,
        arguments.model_input_size,
        arguments.confidence_threshold,
        arguments.nms_threshold,
    )
    if any(option is not None for option in detector_options) and not analyze_objects:
        parser.error("detector options require --analyze or CCTV_DETECTOR_ENABLED=true")
    match_faces = arguments.match_faces or settings.face_matching_enabled
    face_match_options = (arguments.face_match_threshold, arguments.face_match_margin)
    if any(option is not None for option in face_match_options) and not match_faces:
        parser.error(
            "face matching options require --match-faces or CCTV_FACE_MATCHING_ENABLED=true"
        )
    effective_sample_fps = (
        arguments.sample_fps if arguments.sample_fps is not None else settings.analysis_fps
    )

    configure_logging(
        level=settings.log_level,
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    try:
        frame_consumers: list[FrameConsumer] = []
        detector: ObjectDetector | None = None
        tracker: IoUTracker | None = None
        detection_repository: DetectionRepository | None = None
        rule_engine: RuleEngine | None = None
        emitted_rule_events = 0
        filtered_out_detections = 0
        hiperwall_planner: HiperwallDryRunPlanner | None = None
        simulated_hiperwall_actions = 0
        analysis_run: AnalysisRunRecord | None = None
        analysis_run_finished = False
        face_matching_consumer: FaceMatchingConsumer | None = None
        if analyze_objects:
            detector = create_detector(
                DetectorConfig(
                    detector_type=settings.detector_type,
                    model_path=arguments.model_path or settings.model_path,
                    class_names_path=(arguments.class_names_path or settings.model_classes_path),
                    device=settings.ai_device.value,
                    input_size=arguments.model_input_size or settings.effective_detector_input_size,
                    confidence_threshold=(
                        arguments.confidence_threshold
                        if arguments.confidence_threshold is not None
                        else settings.effective_detector_confidence_threshold
                    ),
                    nms_threshold=(
                        arguments.nms_threshold
                        if arguments.nms_threshold is not None
                        else settings.effective_detector_nms_threshold
                    ),
                )
            )
            if settings.tracking_enabled:
                tracker = IoUTracker(
                    iou_threshold=settings.tracker_iou_threshold,
                    max_missed_frames=settings.tracker_max_missed_frames,
                    max_idle_seconds=settings.tracker_max_idle_seconds,
                    tracked_class_names=settings.tracker_class_name_list,
                )
            if settings.persist_detections:
                initialize_database(settings.database_path)
                detection_repository = DetectionRepository(settings.database_path)
                metadata = detector.metadata
                analysis_run = detection_repository.create_analysis_run(
                    source_type="local_video",
                    source_name=Path(video_path).name,
                    detector_type=metadata.detector_type,
                    model_name=metadata.model_name,
                    model_sha256=metadata.model_sha256,
                    device=metadata.device,
                    input_size=metadata.input_size,
                    confidence_threshold=metadata.confidence_threshold,
                    nms_threshold=metadata.nms_threshold,
                    sample_fps=effective_sample_fps,
                )
                if settings.rules_enabled and tracker is not None:
                    definitions = RuleRepository(settings.database_path).list_rule_definitions(
                        source_name=Path(video_path).name
                    )
                    rule_engine = RuleEngine(definitions)
                    logger.info(
                        "Rule engine initialized",
                        extra={
                            "event": "rule_engine_initialized",
                            "source_name": Path(video_path).name,
                            "rule_count": len(definitions),
                            "supported_rule_types": rule_engine.supported_rule_types,
                        },
                    )
                    if settings.hiperwall_dry_run_enabled:
                        if settings.app_mode is AppMode.DRY_RUN:
                            hiperwall_planner = HiperwallDryRunPlanner(settings)
                        else:
                            logger.warning(
                                "Hiperwall DRY RUN planning is unavailable in LIVE mode",
                                extra={
                                    "event": "hiperwall_dry_run_unavailable",
                                    "app_mode": settings.app_mode,
                                },
                            )
            if settings.rules_enabled and rule_engine is None:
                logger.warning(
                    "Rule engine disabled for this run because tracking or persistence is unavailable",
                    extra={
                        "event": "rule_engine_unavailable",
                        "tracking_enabled": tracker is not None,
                        "persistence_enabled": settings.persist_detections,
                    },
                )

            def analyze_frame(frame: DecodedFrame) -> None:
                nonlocal emitted_rule_events
                nonlocal filtered_out_detections
                nonlocal simulated_hiperwall_actions

                raw_result = detector.analyze(frame)
                result = filter_detections_by_class(
                    raw_result,
                    settings.detection_class_name_list,
                )
                filtered_out_detections += len(raw_result.detections) - len(
                    result.detections
                )
                if tracker is not None:
                    result = tracker.update(result)
                    if face_matching_consumer is not None:
                        face_matching_consumer.process_tracked(
                            frame,
                            result,
                            active_track_ids=tracker.active_track_ids,
                        )
                rule_events = rule_engine.process(result) if rule_engine is not None else ()
                emitted_rule_events += len(rule_events)
                display_actions = (
                    hiperwall_planner.plan(
                        rule_events,
                        analysis_run_id=analysis_run.id,
                        source_name=Path(video_path).name,
                    )
                    if hiperwall_planner is not None and analysis_run is not None
                    else ()
                )
                simulated_hiperwall_actions += len(display_actions)
                if detection_repository is not None and analysis_run is not None:
                    detection_repository.save_frame(
                        analysis_run.id,
                        result,
                        active_track_ids=(
                            tracker.active_track_ids if tracker is not None else None
                        ),
                        rule_events=rule_events,
                        display_actions=display_actions,
                    )

            frame_consumers.append(analyze_frame)

        if match_faces:
            observation_sink = (
                create_resolving_face_observation_sink(
                    settings,
                    analysis_run_id=analysis_run.id,
                )
                if analysis_run is not None
                else None
            )
            face_matching_consumer = create_sface_matching_consumer(
                settings,
                source_name=Path(video_path).name,
                similarity_threshold=arguments.face_match_threshold,
                minimum_margin=arguments.face_match_margin,
                observation_sink=observation_sink,
            )
            if tracker is None:
                frame_consumers.append(face_matching_consumer)
            else:
                logger.info(
                    "Track-aware face matching cache enabled",
                    extra={
                        "event": "track_face_matching_cache_enabled",
                        "source_name": Path(video_path).name,
                        "unknown_retry_seconds": settings.face_match_unknown_retry_seconds,
                    },
                )

        snapshot_writer: SnapshotWriter | None = None
        if save_snapshots:
            snapshot_root = arguments.snapshot_dir or settings.snapshot_dir
            snapshot_writer = SnapshotWriter(
                build_snapshot_run_directory(snapshot_root, video_path),
                jpeg_quality=(
                    arguments.jpeg_quality
                    if arguments.jpeg_quality is not None
                    else settings.snapshot_jpeg_quality
                ),
            )
            frame_consumers.append(snapshot_writer)

        frame_consumer: FrameConsumer | None = None
        if len(frame_consumers) == 1:
            frame_consumer = frame_consumers[0]
        elif frame_consumers:
            frame_consumer = SequentialFrameConsumer(*frame_consumers)

        try:
            result = LocalVideoWorker(
                video_path,
                sample_fps=effective_sample_fps,
                frame_consumer=frame_consumer,
                max_samples=arguments.max_samples,
            ).execute()
            if detection_repository is not None and analysis_run is not None:
                analysis_run = detection_repository.finish_analysis_run(
                    analysis_run.id,
                    status=AnalysisRunStatus(result.status.value),
                )
                analysis_run_finished = True
        except Exception as error:
            if (
                detection_repository is not None
                and analysis_run is not None
                and not analysis_run_finished
            ):
                try:
                    detection_repository.finish_analysis_run(
                        analysis_run.id,
                        status=AnalysisRunStatus.FAILED,
                        error_type=type(error).__name__,
                    )
                except Exception:
                    logger.exception(
                        "Analysis run failure state could not be persisted",
                        extra={
                            "event": "analysis_run_failure_persistence_failed",
                            "analysis_run_id": analysis_run.id,
                        },
                    )
            raise
        output = asdict(result)
        if snapshot_writer is not None:
            output["snapshots"] = {
                "output_dir": snapshot_writer.output_dir,
                "saved_count": snapshot_writer.saved_count,
                "first_path": snapshot_writer.first_path,
                "last_path": snapshot_writer.last_path,
                "jpeg_quality": snapshot_writer.jpeg_quality,
            }
            logger.info(
                "Snapshot run complete",
                extra={
                    "event": "snapshot_run_completed",
                    "snapshot_dir": snapshot_writer.output_dir,
                    "saved_count": snapshot_writer.saved_count,
                    "first_snapshot_path": snapshot_writer.first_path,
                    "last_snapshot_path": snapshot_writer.last_path,
                    "jpeg_quality": snapshot_writer.jpeg_quality,
                },
            )
        if detector is not None:
            output["analysis"] = asdict(detector.summary)
            output["analysis"]["persistence_enabled"] = settings.persist_detections
            output["analysis"]["analysis_run_id"] = (
                analysis_run.id if analysis_run is not None else None
            )
            output["analysis"]["tracking_enabled"] = tracker is not None
            output["analysis"]["detection_class_names"] = (
                settings.detection_class_name_list
            )
            output["analysis"]["tracker_class_names"] = settings.tracker_class_name_list
            output["analysis"]["filtered_out_detections"] = filtered_out_detections
            output["analysis"]["tracking"] = (
                asdict(tracker.summary) if tracker is not None else None
            )
            output["analysis"]["rules"] = {
                "configured": settings.rules_enabled,
                "enabled": rule_engine is not None,
                "loaded_count": len(rule_engine.rules) if rule_engine is not None else 0,
                "emitted_event_count": emitted_rule_events,
            }
            output["analysis"]["hiperwall"] = {
                "dry_run_configured": settings.hiperwall_dry_run_enabled,
                "dry_run_enabled": hiperwall_planner is not None,
                "simulated_action_count": simulated_hiperwall_actions,
                "external_request_sent": False,
            }
            logger.info(
                "Object detection run complete",
                extra={
                    "event": "detector_run_completed",
                    "analysis_run_id": analysis_run.id if analysis_run is not None else None,
                    "persistence_enabled": settings.persist_detections,
                    "tracking_enabled": tracker is not None,
                    "detection_class_names": settings.detection_class_name_list,
                    "tracker_class_names": settings.tracker_class_name_list,
                    "filtered_out_detections": filtered_out_detections,
                    "tracking": asdict(tracker.summary) if tracker is not None else None,
                    "rules_enabled": rule_engine is not None,
                    "loaded_rule_count": len(rule_engine.rules) if rule_engine is not None else 0,
                    "emitted_rule_event_count": emitted_rule_events,
                    "simulated_hiperwall_action_count": simulated_hiperwall_actions,
                    **asdict(detector.summary),
                },
            )
        if face_matching_consumer is not None:
            output["face_matching"] = asdict(face_matching_consumer.summary)
            logger.info(
                "Face matching run complete",
                extra={
                    "event": "face_matching_run_completed",
                    "source_name": Path(video_path).name,
                    **asdict(face_matching_consumer.summary),
                },
            )
        print(json.dumps(output, default=str, ensure_ascii=False, sort_keys=True))
    finally:
        shutdown_logging()
