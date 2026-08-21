"""Continuously reconcile registered cameras with owned RTSP analysis workers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Protocol
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

from cctv.core.settings import Settings
from cctv.db import (
    AnalysisLeaseRepository,
    CameraProvisioningStatus,
    CameraRecord,
    CameraRepository,
    RuleRecord,
    RuleRepository,
)

logger = logging.getLogger(__name__)


class AnalysisWorkerStatus(StrEnum):
    """Credential-free lifecycle state exposed by the supervisor API."""

    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class AnalysisWorkerSnapshot:
    camera_id: int
    camera_name: str
    stream_path: str
    status: AnalysisWorkerStatus
    desired: bool
    lease_owned: bool
    rule_count: int
    configuration_revision: str | None
    analysis_run_id: str | None
    processed_samples: int
    connection_count: int
    reconnect_count: int
    restart_count: int
    started_at: datetime | None
    last_frame_at: datetime | None
    last_event_at: datetime | None
    completed_at: datetime | None
    next_restart_at: datetime | None
    last_error_type: str | None
    last_message: str | None
    requested_device: str | None = None
    effective_device: str | None = None
    execution_provider: str | None = None
    cpu_fallback: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisSupervisorSnapshot:
    enabled: bool
    running: bool
    max_workers: int
    active_workers: int
    last_reconciled_at: datetime | None
    items: tuple[AnalysisWorkerSnapshot, ...]


class ManagedProcess(Protocol):
    pid: int
    stdout: Iterable[str] | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[[list[str], dict[str, str]], ManagedProcess]
Clock = Callable[[], float]
NowFactory = Callable[[], datetime]


@dataclass(slots=True)
class _ManagedWorker:
    camera_id: int
    camera_name: str
    stream_path: str
    status: AnalysisWorkerStatus = AnalysisWorkerStatus.STOPPED
    desired: bool = False
    lease_owned: bool = False
    rule_count: int = 0
    configuration_revision: str | None = None
    process: ManagedProcess | None = None
    analysis_run_id: str | None = None
    processed_samples: int = 0
    connection_count: int = 0
    reconnect_count: int = 0
    restart_count: int = 0
    started_at: datetime | None = None
    last_frame_at: datetime | None = None
    last_event_at: datetime | None = None
    completed_at: datetime | None = None
    next_restart_at: datetime | None = None
    next_restart_monotonic: float = 0.0
    last_error_type: str | None = None
    last_message: str | None = None
    requested_device: str | None = None
    effective_device: str | None = None
    execution_provider: str | None = None
    cpu_fallback: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _DesiredWorker:
    camera: CameraRecord
    rules: tuple[RuleRecord, ...]
    revision: str


class AnalysisSupervisor:
    """Own at most one continuously running analysis process per eligible camera."""

    def __init__(
        self,
        settings: Settings,
        *,
        camera_repository: CameraRepository | None = None,
        rule_repository: RuleRepository | None = None,
        lease_repository: AnalysisLeaseRepository | None = None,
        process_factory: ProcessFactory | None = None,
        owner_id: str | None = None,
        monotonic_clock: Clock = monotonic,
        now_factory: NowFactory | None = None,
    ) -> None:
        self.settings = settings
        self.camera_repository = camera_repository or CameraRepository(settings.database_path)
        self.rule_repository = rule_repository or RuleRepository(settings.database_path)
        self.lease_repository = lease_repository or AnalysisLeaseRepository(settings.database_path)
        self._process_factory = process_factory or _popen
        self._owner_id = owner_id or str(uuid4())
        self._monotonic = monotonic_clock
        self._now = now_factory or (lambda: datetime.now(UTC))
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._workers: dict[int, _ManagedWorker] = {}
        self._last_reconciled_at: datetime | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start one idempotent reconciliation thread."""
        with self._lock:
            if self._thread is not None:
                return
            self._thread = Thread(
                target=self._run,
                name="analysis-supervisor",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            "Analysis supervisor started",
            extra={
                "event": "analysis_supervisor_started",
                "max_workers": self.settings.analysis_supervisor_max_workers,
                "reconcile_interval_seconds": (
                    self.settings.analysis_supervisor_reconcile_interval_seconds
                ),
            },
        )

    def stop(self) -> None:
        """Stop reconciliation and terminate every owned worker."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(
                timeout=(
                    self.settings.analysis_supervisor_reconcile_interval_seconds
                    + self.settings.analysis_supervisor_shutdown_timeout_seconds
                    + 1
                )
            )
        self._stop_all("supervisor_shutdown")
        logger.info(
            "Analysis supervisor stopped",
            extra={"event": "analysis_supervisor_stopped"},
        )

    def reconcile_once(self) -> None:
        """Apply one deterministic desired-state cycle; public for diagnostics and tests."""
        self._observe_exited_processes()
        desired = self._load_desired_workers()
        desired_by_id = {item.camera.id: item for item in desired}

        with self._lock:
            known_ids = set(self._workers)
        for removed_id in known_ids - set(desired_by_id):
            self._mark_not_desired(removed_id, "camera disabled or has no enabled rules")

        for item in desired:
            entry = self._entry_for(item.camera)
            needs_restart = False
            with self._lock:
                entry.camera_name = item.camera.name
                entry.stream_path = item.camera.stream_path
                entry.desired = True
                entry.rule_count = len(item.rules)
                if entry.process is not None and entry.configuration_revision != item.revision:
                    needs_restart = True
                entry.configuration_revision = item.revision
            if needs_restart:
                self._stop_worker(entry, "configuration_changed")

        self._renew_owned_leases()
        self._start_available_workers(desired)
        with self._lock:
            self._last_reconciled_at = self._now()

    def snapshot(self) -> AnalysisSupervisorSnapshot:
        with self._lock:
            items = tuple(
                self._worker_snapshot(item)
                for item in sorted(self._workers.values(), key=lambda value: value.camera_id)
            )
            return AnalysisSupervisorSnapshot(
                enabled=True,
                running=self._thread is not None and self._thread.is_alive(),
                max_workers=self.settings.analysis_supervisor_max_workers,
                active_workers=sum(item.process is not None for item in self._workers.values()),
                last_reconciled_at=self._last_reconciled_at,
                items=items,
            )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self.reconcile_once()
                except Exception:
                    logger.exception(
                        "Analysis supervisor reconciliation failed",
                        extra={"event": "analysis_supervisor_reconciliation_failed"},
                    )
                self._stop_event.wait(self.settings.analysis_supervisor_reconcile_interval_seconds)
        finally:
            self._stop_all("supervisor_shutdown")

    def _load_desired_workers(self) -> tuple[_DesiredWorker, ...]:
        if not self.settings.rules_enabled:
            return ()
        cameras = self.camera_repository.list_cameras(page=1, limit=100).items
        rules_page = self.rule_repository.list_rules(page=1, limit=100, enabled=True)
        if rules_page.total > 100:
            raise RuntimeError("analysis supervisor cannot load more than 100 enabled rules")
        rules_by_source: dict[str, list[RuleRecord]] = {}
        for rule in rules_page.items:
            rules_by_source.setdefault(rule.source_name.casefold(), []).append(rule)
        eligible_statuses = {
            CameraProvisioningStatus.ACTIVE,
            CameraProvisioningStatus.EXTERNAL,
        }
        desired: list[_DesiredWorker] = []
        for camera in sorted(cameras, key=lambda item: item.id):
            rules = tuple(rules_by_source.get(camera.stream_path.casefold(), ()))
            if (
                not camera.enabled
                or camera.provisioning_status not in eligible_statuses
                or not rules
            ):
                continue
            desired.append(
                _DesiredWorker(
                    camera=camera,
                    rules=rules,
                    revision=_configuration_revision(camera, rules),
                )
            )
        return tuple(desired)

    def _entry_for(self, camera: CameraRecord) -> _ManagedWorker:
        with self._lock:
            entry = self._workers.get(camera.id)
            if entry is None:
                entry = _ManagedWorker(
                    camera_id=camera.id,
                    camera_name=camera.name,
                    stream_path=camera.stream_path,
                )
                self._workers[camera.id] = entry
            return entry

    def _mark_not_desired(self, camera_id: int, message: str) -> None:
        with self._lock:
            entry = self._workers.get(camera_id)
            if entry is None:
                return
            entry.desired = False
            entry.rule_count = 0
            entry.configuration_revision = None
            has_process = entry.process is not None
        if has_process:
            self._stop_worker(entry, "no_longer_desired", release_lease=True)
        else:
            self._release_lease(entry)
        with self._lock:
            entry.status = AnalysisWorkerStatus.STOPPED
            entry.last_message = message
            entry.next_restart_at = None
            entry.next_restart_monotonic = 0.0

    def _start_available_workers(self, desired: tuple[_DesiredWorker, ...]) -> None:
        for item in desired:
            with self._lock:
                entry = self._workers[item.camera.id]
                active_count = sum(worker.process is not None for worker in self._workers.values())
                ready = (
                    entry.process is None
                    and self._monotonic() >= entry.next_restart_monotonic
                    and active_count < self.settings.analysis_supervisor_max_workers
                )
                if not ready:
                    if (
                        entry.process is None
                        and active_count >= self.settings.analysis_supervisor_max_workers
                    ):
                        entry.status = AnalysisWorkerStatus.STOPPED
                        entry.last_message = "waiting for worker capacity"
                    continue
            if not self._ensure_lease(entry):
                with self._lock:
                    entry.status = AnalysisWorkerStatus.STOPPED
                    entry.last_message = "another Backend instance owns this camera"
                continue
            self._start_worker(entry)

    def _renew_owned_leases(self) -> None:
        with self._lock:
            entries = tuple(entry for entry in self._workers.values() if entry.lease_owned)
        for entry in entries:
            if self._ensure_lease(entry):
                continue
            with self._lock:
                has_process = entry.process is not None
                entry.last_message = "camera ownership lease was lost"
            if has_process:
                self._stop_worker(entry, "lease_lost")

    def _ensure_lease(self, entry: _ManagedWorker) -> bool:
        acquired = self.lease_repository.acquire(
            entry.camera_id,
            self._owner_id,
            lease_seconds=self.settings.analysis_supervisor_lease_seconds,
            now=self._now(),
        )
        with self._lock:
            entry.lease_owned = acquired
        return acquired

    def _release_lease(self, entry: _ManagedWorker) -> None:
        with self._lock:
            owned = entry.lease_owned
            entry.lease_owned = False
        if owned:
            self.lease_repository.release(entry.camera_id, self._owner_id)

    def _start_worker(self, entry: _ManagedWorker) -> None:
        if not self.settings.model_path.is_file():
            self._record_start_failure(entry, "FileNotFoundError", "detector model is missing")
            return
        command = self._build_command(entry)
        environment = self._worker_environment(entry)
        try:
            process = self._process_factory(command, environment)
        except Exception as error:
            self._record_start_failure(entry, type(error).__name__, "worker process start failed")
            logger.exception(
                "Analysis worker process could not be started",
                extra={
                    "event": "analysis_worker_start_failed",
                    "camera_id": entry.camera_id,
                    "source_name": entry.stream_path,
                    "error_type": type(error).__name__,
                },
            )
            return

        with self._lock:
            entry.process = process
            entry.status = AnalysisWorkerStatus.STARTING
            entry.analysis_run_id = None
            entry.processed_samples = 0
            entry.connection_count = 0
            entry.reconnect_count = 0
            entry.started_at = self._now()
            entry.completed_at = None
            entry.next_restart_at = None
            entry.next_restart_monotonic = 0.0
            entry.last_error_type = None
            entry.last_message = "analysis worker process started"
            entry.requested_device = self.settings.ai_device.value
            entry.effective_device = None
            entry.execution_provider = None
            entry.cpu_fallback = False
            entry.fallback_reason = None
        Thread(
            target=self._read_worker_output,
            args=(entry.camera_id, process),
            name=f"analysis-output-{entry.camera_id}",
            daemon=True,
        ).start()
        logger.info(
            "Analysis worker started",
            extra={
                "event": "analysis_worker_started",
                "camera_id": entry.camera_id,
                "source_name": entry.stream_path,
                "rule_count": entry.rule_count,
                "restart_count": entry.restart_count,
            },
        )

    def _record_start_failure(
        self,
        entry: _ManagedWorker,
        error_type: str,
        message: str,
    ) -> None:
        now = self._now()
        with self._lock:
            entry.status = AnalysisWorkerStatus.FAILED
            entry.last_error_type = error_type
            entry.last_message = message
            entry.completed_at = now
            entry.restart_count += 1
            delay = self._restart_delay(entry.restart_count)
            entry.next_restart_monotonic = self._monotonic() + delay
            entry.next_restart_at = now + timedelta(seconds=delay)

    def _observe_exited_processes(self) -> None:
        with self._lock:
            active = tuple(
                (entry, entry.process)
                for entry in self._workers.values()
                if entry.process is not None
            )
        for entry, process in active:
            if process is None:
                continue
            return_code = process.poll()
            if return_code is None:
                continue
            now = self._now()
            with self._lock:
                if entry.process is not process:
                    continue
                entry.process = None
                entry.status = AnalysisWorkerStatus.FAILED
                entry.completed_at = now
                entry.restart_count += 1
                entry.last_error_type = entry.last_error_type or f"ProcessExit{return_code}"
                entry.last_message = "analysis worker exited unexpectedly"
                delay = self._restart_delay(entry.restart_count)
                entry.next_restart_monotonic = self._monotonic() + delay
                entry.next_restart_at = now + timedelta(seconds=delay)
            logger.warning(
                "Analysis worker exited unexpectedly",
                extra={
                    "event": "analysis_worker_exited",
                    "camera_id": entry.camera_id,
                    "source_name": entry.stream_path,
                    "return_code": return_code,
                    "restart_count": entry.restart_count,
                },
            )

    def _stop_worker(
        self,
        entry: _ManagedWorker,
        reason: str,
        *,
        release_lease: bool = False,
    ) -> None:
        with self._lock:
            process = entry.process
            if process is None:
                return
            entry.status = AnalysisWorkerStatus.STOPPING
            entry.last_message = reason
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.settings.analysis_supervisor_shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            with self._lock:
                if entry.process is process:
                    entry.process = None
                    entry.status = AnalysisWorkerStatus.STOPPED
                    entry.completed_at = self._now()
                    entry.next_restart_at = None
                    entry.next_restart_monotonic = 0.0
            logger.info(
                "Analysis worker stopped",
                extra={
                    "event": "analysis_worker_stopped",
                    "camera_id": entry.camera_id,
                    "source_name": entry.stream_path,
                    "reason": reason,
                },
            )
            if release_lease:
                self._release_lease(entry)

    def _stop_all(self, reason: str) -> None:
        with self._lock:
            entries = tuple(self._workers.values())
        for entry in entries:
            self._stop_worker(entry, reason, release_lease=True)
        self.lease_repository.release_all(self._owner_id)

    def _read_worker_output(self, camera_id: int, process: ManagedProcess) -> None:
        if process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                self._consume_worker_record(camera_id, process, record)

    def _consume_worker_record(
        self,
        camera_id: int,
        process: ManagedProcess,
        record: dict[str, Any],
    ) -> None:
        event = record.get("event")
        with self._lock:
            entry = self._workers.get(camera_id)
            if entry is None or entry.process is not process:
                return
            if event == "analysis_run_created":
                entry.analysis_run_id = _optional_text(record.get("analysis_run_id"))
            elif event == "detector_runtime_resolved":
                entry.requested_device = _optional_text(record.get("requested_device"))
                entry.effective_device = _optional_text(record.get("effective_device"))
                entry.execution_provider = _optional_text(record.get("execution_provider"))
                entry.cpu_fallback = record.get("cpu_fallback") is True
                entry.fallback_reason = _optional_text(record.get("fallback_reason"))
            elif event == "rtsp_worker_started":
                entry.status = AnalysisWorkerStatus.STARTING
                entry.last_message = "waiting for RTSP stream"
            elif event == "rtsp_source_connected":
                entry.status = AnalysisWorkerStatus.RUNNING
                entry.connection_count = _integer(record.get("connection_count"))
                entry.last_message = "RTSP stream connected"
            elif event == "rtsp_source_reconnect_scheduled":
                entry.status = AnalysisWorkerStatus.RECONNECTING
                entry.reconnect_count = _integer(record.get("reconnect_count"))
                entry.last_message = "RTSP reconnect scheduled"
            elif event == "rtsp_worker_progress":
                entry.status = AnalysisWorkerStatus.RUNNING
                entry.processed_samples = _integer(record.get("processed_samples"))
                entry.last_frame_at = self._now()
            elif event == "rule_event_emitted":
                entry.last_event_at = self._now()
            elif event == "rtsp_worker_failed":
                entry.last_error_type = _optional_text(record.get("error_type")) or "WorkerError"
                entry.last_message = "RTSP worker failed"
            elif event == "rtsp_worker_finished":
                entry.processed_samples = _integer(record.get("processed_samples"))
                entry.connection_count = _integer(record.get("connection_count"))
                entry.reconnect_count = _integer(record.get("reconnect_count"))

    def _build_command(self, entry: _ManagedWorker) -> list[str]:
        command = [
            sys.executable,
            "-c",
            "from cctv.workers.rtsp import run; run()",
            "--source-name",
            entry.stream_path,
            "--sample-fps",
            str(self.settings.analysis_fps),
            "--analyze",
            "--emit-progress",
        ]
        if self.settings.face_matching_enabled:
            command.append("--match-faces")
        return command

    def _worker_environment(self, entry: _ManagedWorker) -> dict[str, str]:
        environment = os.environ.copy()
        for key in (
            "HIPERWALL_BASE_URL",
            "HIPERWALL_AUTH_MODE",
            "HIPERWALL_USER",
            "HIPERWALL_TOKEN",
            "CCTV_CAMERA_CREDENTIAL_KEY",
        ):
            environment.pop(key, None)
        log_path = (
            self.settings.log_path.parent / f"analysis-camera-{entry.camera_id}.jsonl"
        ).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "CCTV_APP_ENV": self.settings.app_env,
                "CCTV_APP_MODE": self.settings.app_mode.value,
                "CCTV_AI_DEVICE": self.settings.ai_device.value,
                "CCTV_AI_ALLOW_CPU_FALLBACK": str(
                    self.settings.ai_allow_cpu_fallback
                ).lower(),
                "CCTV_AI_CUDA_DEVICE_ID": str(self.settings.ai_cuda_device_id),
                "CCTV_ANALYSIS_FPS": str(self.settings.analysis_fps),
                "CCTV_ANALYSIS_SUPERVISOR_ENABLED": "false",
                "CCTV_DATABASE_PATH": str(self.settings.database_path),
                "CCTV_MODEL_PATH": str(self.settings.model_path),
                "CCTV_DETECTOR_ENABLED": "true",
                "CCTV_DETECTOR_TYPE": self.settings.detector_type,
                "CCTV_DETECTOR_INPUT_SIZE": str(self.settings.effective_detector_input_size),
                "CCTV_DETECTOR_CONFIDENCE_THRESHOLD": str(
                    self.settings.effective_detector_confidence_threshold
                ),
                "CCTV_DETECTOR_NMS_THRESHOLD": str(self.settings.effective_detector_nms_threshold),
                "CCTV_DETECTION_CLASS_NAMES": self.settings.detection_class_names,
                "CCTV_TRACKING_ENABLED": str(self.settings.tracking_enabled).lower(),
                "CCTV_TRACKER_IOU_THRESHOLD": str(self.settings.tracker_iou_threshold),
                "CCTV_TRACKER_MAX_MISSED_FRAMES": str(self.settings.tracker_max_missed_frames),
                "CCTV_TRACKER_MAX_IDLE_SECONDS": str(self.settings.tracker_max_idle_seconds),
                "CCTV_TRACKER_CLASS_NAMES": self.settings.tracker_class_names,
                "CCTV_PERSIST_DETECTIONS": "true",
                "CCTV_RULES_ENABLED": "true",
                "CCTV_HIPERWALL_EXECUTOR_ENABLED": "false",
                "CCTV_HIPERWALL_DRY_RUN_ENABLED": str(
                    self.settings.hiperwall_dry_run_enabled
                ).lower(),
                "HIPERWALL_DEFAULT_DISPLAY_SECONDS": str(
                    self.settings.hiperwall_default_display_seconds
                ),
                "CCTV_FACE_MATCHING_ENABLED": str(self.settings.face_matching_enabled).lower(),
                "CCTV_FACE_DETECTION_MODEL_PATH": str(self.settings.face_detection_model_path),
                "CCTV_FACE_EMBEDDING_MODEL_PATH": str(self.settings.face_embedding_model_path),
                "CCTV_FACE_MATCH_SIMILARITY_THRESHOLD": str(
                    self.settings.face_match_similarity_threshold
                ),
                "CCTV_FACE_MATCH_MINIMUM_MARGIN": str(self.settings.face_match_minimum_margin),
                "CCTV_FACE_MATCH_UNKNOWN_RETRY_SECONDS": str(
                    self.settings.face_match_unknown_retry_seconds
                ),
                "CCTV_RTSP_WORKER_URL": camera_rtsp_url(
                    self.settings.rtsp_worker_url,
                    entry.stream_path,
                ),
                "CCTV_RTSP_SOURCE_NAME": entry.stream_path,
                "CCTV_RTSP_OPEN_TIMEOUT_SECONDS": str(self.settings.rtsp_open_timeout_seconds),
                "CCTV_RTSP_READ_TIMEOUT_SECONDS": str(self.settings.rtsp_read_timeout_seconds),
                "CCTV_RTSP_RECONNECT_INITIAL_SECONDS": str(
                    self.settings.rtsp_reconnect_initial_seconds
                ),
                "CCTV_RTSP_RECONNECT_MAX_SECONDS": str(self.settings.rtsp_reconnect_max_seconds),
                "CCTV_RTSP_RECONNECT_JITTER_RATIO": str(self.settings.rtsp_reconnect_jitter_ratio),
                "CCTV_RTSP_MAX_RETRIES": str(self.settings.rtsp_max_retries),
                "CCTV_LOG_LEVEL": self.settings.log_level,
                "CCTV_LOG_PATH": str(log_path),
            }
        )
        if self.settings.model_classes_path is None:
            environment.pop("CCTV_MODEL_CLASSES_PATH", None)
        else:
            environment["CCTV_MODEL_CLASSES_PATH"] = str(self.settings.model_classes_path)
        if self.settings.ai_cuda_gpu_mem_limit_mb is None:
            environment.pop("CCTV_AI_CUDA_GPU_MEM_LIMIT_MB", None)
        else:
            environment["CCTV_AI_CUDA_GPU_MEM_LIMIT_MB"] = str(
                self.settings.ai_cuda_gpu_mem_limit_mb
            )
        return environment

    def _restart_delay(self, restart_count: int) -> float:
        return min(
            self.settings.analysis_supervisor_restart_max_seconds,
            self.settings.analysis_supervisor_restart_base_seconds
            * (2 ** max(restart_count - 1, 0)),
        )

    @staticmethod
    def _worker_snapshot(entry: _ManagedWorker) -> AnalysisWorkerSnapshot:
        return AnalysisWorkerSnapshot(
            camera_id=entry.camera_id,
            camera_name=entry.camera_name,
            stream_path=entry.stream_path,
            status=entry.status,
            desired=entry.desired,
            lease_owned=entry.lease_owned,
            rule_count=entry.rule_count,
            configuration_revision=entry.configuration_revision,
            analysis_run_id=entry.analysis_run_id,
            processed_samples=entry.processed_samples,
            connection_count=entry.connection_count,
            reconnect_count=entry.reconnect_count,
            restart_count=entry.restart_count,
            started_at=entry.started_at,
            last_frame_at=entry.last_frame_at,
            last_event_at=entry.last_event_at,
            completed_at=entry.completed_at,
            next_restart_at=entry.next_restart_at,
            last_error_type=entry.last_error_type,
            last_message=entry.last_message,
            requested_device=entry.requested_device,
            effective_device=entry.effective_device,
            execution_provider=entry.execution_provider,
            cpu_fallback=entry.cpu_fallback,
            fallback_reason=entry.fallback_reason,
        )


def camera_rtsp_url(configured_url: str, stream_path: str) -> str:
    """Replace the configured template path with one validated camera stream path."""
    parsed = urlsplit(configured_url)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"/{quote(stream_path, safe='/')}",
            "",
            "",
        )
    )


def _configuration_revision(camera: CameraRecord, rules: tuple[RuleRecord, ...]) -> str:
    payload = {
        "camera": {
            "id": camera.id,
            "stream_path": camera.stream_path,
            "enabled": camera.enabled,
            "provisioning_status": camera.provisioning_status,
        },
        "rules": [asdict(rule) for rule in rules],
    }
    encoded = json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _popen(command: list[str], environment: dict[str, str]) -> ManagedProcess:
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


__all__ = [
    "AnalysisSupervisor",
    "AnalysisSupervisorSnapshot",
    "AnalysisWorkerSnapshot",
    "AnalysisWorkerStatus",
    "camera_rtsp_url",
]
