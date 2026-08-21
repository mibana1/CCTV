"""Long-running RTSP sampling and interchangeable object-detection worker."""

from __future__ import annotations

import argparse
import json
import logging
import signal
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from types import FrameType

from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import AppMode, get_settings
from cctv.db import (
    AnalysisRunRecord,
    AnalysisRunStatus,
    DetectionRepository,
    RuleRepository,
    initialize_database,
)
from cctv.hiperwall import HiperwallDryRunPlanner, HiperwallLivePlanner
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
    RtspStreamReader,
    SnapshotWriter,
    build_snapshot_run_directory,
)
from cctv.rules import RuleEngine
from cctv.vision import UpperBodyColorAnalyzer
from cctv.workers.local_video import (
    FrameConsumer,
    SequentialFrameConsumer,
    WorkerStatus,
    confidence_threshold,
    jpeg_quality,
    model_input_size,
    nms_threshold,
    positive_integer,
)

logger = logging.getLogger(__name__)
_TIMESTAMP_EPSILON_SECONDS = 1e-9


class RtspStopReason(StrEnum):
    """Reason an RTSP worker ended without a processing failure."""

    SAMPLE_LIMIT = "sample_limit"
    STOP_REQUESTED = "stop_requested"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class RtspWorkerResult:
    """Credential-free summary returned when an RTSP worker stops."""

    status: WorkerStatus
    stop_reason: RtspStopReason
    source_name: str
    sample_fps: float
    processed_samples: int
    first_source_index: int | None
    last_source_index: int | None
    first_timestamp_seconds: float | None
    last_timestamp_seconds: float | None
    elapsed_seconds: float
    connection_count: int
    reconnect_count: int
    decoded_frames: int


class RtspWorker:
    """Sample a reconnecting RTSP reader and dispatch frames synchronously."""

    def __init__(
        self,
        reader: RtspStreamReader,
        *,
        sample_fps: float,
        frame_consumer: FrameConsumer | None = None,
        max_samples: int | None = None,
        emit_progress: bool = False,
    ) -> None:
        if not isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be a finite number greater than zero")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be greater than zero when provided")

        self.reader = reader
        self.sample_fps = float(sample_fps)
        self.frame_consumer = frame_consumer
        self.max_samples = max_samples
        self.emit_progress = emit_progress
        self._status = WorkerStatus.IDLE
        self._status_lock = Lock()
        self._stop_event = Event()

    @property
    def status(self) -> WorkerStatus:
        """Return the current lifecycle state in a thread-safe manner."""
        with self._status_lock:
            return self._status

    def stop(self) -> None:
        """Request cooperative shutdown, including during reconnect waits."""
        self._stop_event.set()
        logger.info(
            "RTSP worker stop requested",
            extra={
                "event": "rtsp_worker_stop_requested",
                "source_name": self.reader.source_name,
            },
        )

    def execute(self) -> RtspWorkerResult:
        """Consume, sample, and dispatch frames until stopped or bounded for a test."""
        self._begin_execution()
        started_at = perf_counter()
        sample_interval_seconds = 1.0 / self.sample_fps
        next_sample_at = 0.0
        processed_samples = 0
        first_source_index: int | None = None
        last_source_index: int | None = None
        first_timestamp_seconds: float | None = None
        last_timestamp_seconds: float | None = None
        stop_reason = RtspStopReason.STOP_REQUESTED
        frame_iterator = self.reader.frames(stop_event=self._stop_event)

        logger.info(
            "RTSP worker started",
            extra={
                "event": "rtsp_worker_started",
                "source_name": self.reader.source_name,
                "sample_fps": self.sample_fps,
                "max_samples": self.max_samples,
            },
        )
        try:
            for raw_frame in frame_iterator:
                if self._stop_event.is_set():
                    break
                if raw_frame.timestamp_seconds + _TIMESTAMP_EPSILON_SECONDS < next_sample_at:
                    continue

                sampled_frame = DecodedFrame(
                    source_index=raw_frame.source_index,
                    sample_index=processed_samples,
                    timestamp_seconds=raw_frame.timestamp_seconds,
                    image=raw_frame.image,
                )
                if self.frame_consumer is not None:
                    self.frame_consumer(sampled_frame)
                processed_samples += 1
                if self.emit_progress:
                    logger.info(
                        "RTSP worker progress updated",
                        extra={
                            "event": "rtsp_worker_progress",
                            "source_name": self.reader.source_name,
                            "processed_samples": processed_samples,
                            "max_samples": self.max_samples,
                            "source_index": sampled_frame.source_index,
                            "sample_index": sampled_frame.sample_index,
                            "timestamp_seconds": sampled_frame.timestamp_seconds,
                        },
                    )
                if first_source_index is None:
                    first_source_index = sampled_frame.source_index
                    first_timestamp_seconds = sampled_frame.timestamp_seconds
                last_source_index = sampled_frame.source_index
                last_timestamp_seconds = sampled_frame.timestamp_seconds

                while next_sample_at <= raw_frame.timestamp_seconds + _TIMESTAMP_EPSILON_SECONDS:
                    next_sample_at += sample_interval_seconds
                if self.max_samples is not None and processed_samples >= self.max_samples:
                    stop_reason = RtspStopReason.SAMPLE_LIMIT
                    break
            final_status = (
                WorkerStatus.COMPLETED
                if stop_reason is RtspStopReason.SAMPLE_LIMIT
                else WorkerStatus.STOPPED
            )
            self._set_status(final_status)
        except KeyboardInterrupt:
            self._stop_event.set()
            stop_reason = RtspStopReason.INTERRUPTED
            final_status = WorkerStatus.STOPPED
            self._set_status(final_status)
        except Exception:
            self._set_status(WorkerStatus.FAILED)
            logger.exception(
                "RTSP worker failed",
                extra={
                    "event": "rtsp_worker_failed",
                    "source_name": self.reader.source_name,
                    "processed_samples": processed_samples,
                },
            )
            raise
        finally:
            frame_iterator.close()

        statistics = self.reader.statistics
        result = RtspWorkerResult(
            status=final_status,
            stop_reason=stop_reason,
            source_name=self.reader.source_name,
            sample_fps=self.sample_fps,
            processed_samples=processed_samples,
            first_source_index=first_source_index,
            last_source_index=last_source_index,
            first_timestamp_seconds=first_timestamp_seconds,
            last_timestamp_seconds=last_timestamp_seconds,
            elapsed_seconds=round(perf_counter() - started_at, 6),
            connection_count=statistics.connection_count,
            reconnect_count=statistics.reconnect_count,
            decoded_frames=statistics.decoded_frames,
        )
        logger.info(
            "RTSP worker finished",
            extra={
                "event": "rtsp_worker_finished",
                "source_name": result.source_name,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "processed_samples": result.processed_samples,
                "connection_count": result.connection_count,
                "reconnect_count": result.reconnect_count,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
        return result

    def _begin_execution(self) -> None:
        with self._status_lock:
            if self._status is not WorkerStatus.IDLE:
                raise RuntimeError("RTSP worker instances can only be executed once")
            self._status = WorkerStatus.RUNNING

    def _set_status(self, status: WorkerStatus) -> None:
        with self._status_lock:
            self._status = status


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for an RTSP analysis worker."""
    parser = argparse.ArgumentParser(description="Analyze a reconnecting RTSP stream")
    parser.add_argument("--rtsp-url")
    parser.add_argument("--source-name")
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--max-samples", type=positive_integer)
    parser.add_argument("--emit-progress", action="store_true", help=argparse.SUPPRESS)
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
    """Run RTSP ingestion with snapshots and optional persisted detection results."""
    settings = get_settings()
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    rtsp_url = arguments.rtsp_url or settings.rtsp_worker_url
    source_name = arguments.source_name or settings.rtsp_source_name
    max_samples = (
        arguments.max_samples if arguments.max_samples is not None else settings.rtsp_max_samples
    )
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
        service="cctv-rtsp-worker",
    )
    try:
        consumers: list[FrameConsumer] = []
        detector: ObjectDetector | None = None
        tracker: IoUTracker | None = None
        repository: DetectionRepository | None = None
        rule_engine: RuleEngine | None = None
        emitted_rule_events = 0
        filtered_out_detections = 0
        hiperwall_planner: HiperwallDryRunPlanner | HiperwallLivePlanner | None = None
        simulated_hiperwall_actions = 0
        queued_hiperwall_actions = 0
        analysis_run: AnalysisRunRecord | None = None
        analysis_run_finished = False
        face_matching_consumer: FaceMatchingConsumer | None = None
        color_analyzer: UpperBodyColorAnalyzer | None = None
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
                    options=settings.detector_runtime_options,
                )
            )
            detector_metadata = detector.metadata
            logger.info(
                "Detector runtime resolved",
                extra={
                    "event": "detector_runtime_resolved",
                    "requested_device": detector_metadata.requested_device,
                    "effective_device": detector_metadata.device,
                    "execution_provider": detector_metadata.execution_provider,
                    "cpu_fallback": detector_metadata.cpu_fallback,
                    "fallback_reason": detector_metadata.fallback_reason,
                },
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
                repository = DetectionRepository(settings.database_path)
                metadata = detector.metadata
                analysis_run = repository.create_analysis_run(
                    source_type="rtsp",
                    source_name=source_name,
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
                        source_name=source_name
                    )
                    rule_engine = RuleEngine(definitions)
                    if any(rule.rule_type == "visual_color" for rule in definitions):
                        color_analyzer = UpperBodyColorAnalyzer()
                    logger.info(
                        "Rule engine initialized",
                        extra={
                            "event": "rule_engine_initialized",
                            "source_name": source_name,
                            "rule_count": len(definitions),
                            "supported_rule_types": rule_engine.supported_rule_types,
                        },
                    )
                    if settings.app_mode is AppMode.LIVE:
                        hiperwall_planner = HiperwallLivePlanner(settings, definitions)
                    elif settings.hiperwall_dry_run_enabled:
                        hiperwall_planner = HiperwallDryRunPlanner(settings, definitions)
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
                nonlocal queued_hiperwall_actions

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
                color_attributes: dict[int, dict[str, object]] = {}
                if color_analyzer is not None:
                    color_attributes = {
                        track_id: observation.rule_attributes
                        for track_id, observation in color_analyzer.analyze(
                            frame,
                            result,
                        ).items()
                    }
                rule_events = (
                    rule_engine.process(result, attributes_by_track=color_attributes)
                    if rule_engine is not None
                    else ()
                )
                display_actions = (
                    hiperwall_planner.plan(
                        rule_events,
                        analysis_run_id=analysis_run.id,
                        source_name=source_name,
                    )
                    if hiperwall_planner is not None and analysis_run is not None
                    else ()
                )
                if repository is not None and analysis_run is not None:
                    outcome = repository.save_frame_with_outcome(
                        analysis_run.id,
                        result,
                        active_track_ids=(
                            tracker.active_track_ids if tracker is not None else None
                        ),
                        rule_events=rule_events,
                        display_actions=display_actions,
                    )
                    persisted_event_ids = set(outcome.persisted_rule_event_ids)
                    persisted_action_ids = set(outcome.persisted_display_action_ids)
                    rule_events = tuple(
                        event for event in rule_events if event.id in persisted_event_ids
                    )
                    display_actions = tuple(
                        action for action in display_actions if action.id in persisted_action_ids
                    )
                for emitted_event in rule_events:
                    logger.info(
                        "Rule event emitted",
                        extra={
                            "event": "rule_event_emitted",
                            "rule_id": emitted_event.rule_id,
                            "track_id": emitted_event.track_id,
                            "event_type": emitted_event.event_type,
                            "event_state": emitted_event.event_state,
                            "timestamp_seconds": emitted_event.occurred_at_seconds,
                            "class_name": emitted_event.class_name,
                            "confidence": emitted_event.confidence,
                            "payload": emitted_event.payload,
                        },
                    )
                emitted_rule_events += len(rule_events)
                if settings.app_mode is AppMode.LIVE:
                    queued_hiperwall_actions += len(display_actions)
                else:
                    simulated_hiperwall_actions += len(display_actions)

            consumers.append(analyze_frame)

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
                source_name=source_name,
                similarity_threshold=arguments.face_match_threshold,
                minimum_margin=arguments.face_match_margin,
                observation_sink=observation_sink,
            )
            if face_matching_consumer is not None and tracker is None:
                consumers.append(face_matching_consumer)
            elif face_matching_consumer is not None:
                logger.info(
                    "Track-aware face matching cache enabled",
                    extra={
                        "event": "track_face_matching_cache_enabled",
                        "source_name": source_name,
                        "unknown_retry_seconds": settings.face_match_unknown_retry_seconds,
                    },
                )

        snapshot_writer: SnapshotWriter | None = None
        if save_snapshots:
            snapshot_writer = SnapshotWriter(
                arguments.snapshot_dir
                or build_snapshot_run_directory(settings.snapshot_dir, source_name),
                jpeg_quality=(
                    arguments.jpeg_quality
                    if arguments.jpeg_quality is not None
                    else settings.snapshot_jpeg_quality
                ),
            )
            consumers.append(snapshot_writer)

        frame_consumer: FrameConsumer | None = None
        if len(consumers) == 1:
            frame_consumer = consumers[0]
        elif consumers:
            frame_consumer = SequentialFrameConsumer(*consumers)

        reader = RtspStreamReader(
            rtsp_url,
            source_name=source_name,
            open_timeout_seconds=settings.rtsp_open_timeout_seconds,
            read_timeout_seconds=settings.rtsp_read_timeout_seconds,
            reconnect_initial_seconds=settings.rtsp_reconnect_initial_seconds,
            reconnect_max_seconds=settings.rtsp_reconnect_max_seconds,
            reconnect_jitter_ratio=settings.rtsp_reconnect_jitter_ratio,
            max_retries=settings.rtsp_max_retries,
        )
        worker = RtspWorker(
            reader,
            sample_fps=effective_sample_fps,
            frame_consumer=frame_consumer,
            max_samples=max_samples,
            emit_progress=arguments.emit_progress,
        )

        def request_stop(signum: int, frame: FrameType | None) -> None:
            del signum, frame
            worker.stop()

        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
        previous_sigint = signal.signal(signal.SIGINT, request_stop)
        try:
            result = worker.execute()
            if repository is not None and analysis_run is not None:
                analysis_run = repository.finish_analysis_run(
                    analysis_run.id,
                    status=AnalysisRunStatus(result.status.value),
                )
                analysis_run_finished = True
        except Exception as error:
            if repository is not None and analysis_run is not None and not analysis_run_finished:
                try:
                    repository.finish_analysis_run(
                        analysis_run.id,
                        status=AnalysisRunStatus.FAILED,
                        error_type=type(error).__name__,
                    )
                except Exception:
                    logger.exception(
                        "RTSP analysis failure state could not be persisted",
                        extra={
                            "event": "analysis_run_failure_persistence_failed",
                            "analysis_run_id": analysis_run.id,
                        },
                    )
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)

        output = asdict(result)
        if snapshot_writer is not None:
            output["snapshots"] = {
                "output_dir": snapshot_writer.output_dir,
                "saved_count": snapshot_writer.saved_count,
                "first_path": snapshot_writer.first_path,
                "last_path": snapshot_writer.last_path,
                "jpeg_quality": snapshot_writer.jpeg_quality,
            }
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
            output["analysis"]["upper_body_color"] = (
                asdict(color_analyzer.summary) if color_analyzer is not None else None
            )
            output["analysis"]["rules"] = {
                "configured": settings.rules_enabled,
                "enabled": rule_engine is not None,
                "loaded_count": len(rule_engine.rules) if rule_engine is not None else 0,
                "emitted_event_count": emitted_rule_events,
            }
            output["analysis"]["hiperwall"] = {
                "mode": settings.app_mode,
                "dry_run_configured": settings.hiperwall_dry_run_enabled,
                "dry_run_enabled": (
                    settings.app_mode is AppMode.DRY_RUN and hiperwall_planner is not None
                ),
                "simulated_action_count": simulated_hiperwall_actions,
                "queued_live_action_count": queued_hiperwall_actions,
                "external_request_sent": False,
            }
        if face_matching_consumer is not None:
            output["face_matching"] = asdict(face_matching_consumer.summary)
            logger.info(
                "Face matching run complete",
                extra={
                    "event": "face_matching_run_completed",
                    "source_name": source_name,
                    **asdict(face_matching_consumer.summary),
                },
            )
        print(json.dumps(output, default=str, ensure_ascii=False, sort_keys=True))
    finally:
        shutdown_logging()
