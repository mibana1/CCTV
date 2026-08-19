"""Launch and observe one local RTSP validation worker at a time."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

from cctv.core.settings import Settings
from cctv.db import DetectionRepository
from cctv.media import sample_index_from_snapshot_name


class TestSessionStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL_SESSION_STATUSES = {
    TestSessionStatus.COMPLETED,
    TestSessionStatus.STOPPED,
    TestSessionStatus.FAILED,
}


class SessionConflictError(RuntimeError):
    """A new session cannot start while another worker is active."""


class SessionNotFoundError(LookupError):
    """The requested in-memory dashboard session does not exist."""


@dataclass(frozen=True, slots=True)
class DashboardEvent:
    sequence: int
    event: str
    timestamp: datetime
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SnapshotInfo:
    name: str
    sample_index: int | None
    size_bytes: int


@dataclass(frozen=True, slots=True)
class TestSessionSnapshot:
    id: str
    status: TestSessionStatus
    camera_id: int
    stream_path: str
    source_name: str
    sample_fps: float
    max_samples: int
    save_snapshots: bool
    match_faces: bool
    analysis_run_id: str | None
    processed_samples: int
    connection_count: int
    reconnect_count: int
    elapsed_seconds: float
    cpu_percent: float | None
    memory_mib: float | None
    peak_cpu_percent: float | None
    peak_memory_mib: float | None
    snapshot_count: int
    started_at: datetime
    completed_at: datetime | None
    last_message: str | None
    error_type: str | None
    result_summary: dict[str, Any] | None


@dataclass(slots=True)
class _MutableSession:
    id: str
    status: TestSessionStatus
    camera_id: int
    stream_path: str
    source_name: str
    rtsp_url: str
    sample_fps: float
    max_samples: int
    save_snapshots: bool
    match_faces: bool
    snapshot_dir: Path
    started_at: datetime
    started_monotonic: float
    process: subprocess.Popen[str] | None = None
    analysis_run_id: str | None = None
    processed_samples: int = 0
    connection_count: int = 0
    reconnect_count: int = 0
    elapsed_seconds: float = 0.0
    cpu_percent: float | None = None
    memory_mib: float | None = None
    peak_cpu_percent: float | None = None
    peak_memory_mib: float | None = None
    completed_at: datetime | None = None
    last_message: str | None = None
    error_type: str | None = None
    result_summary: dict[str, Any] | None = None
    events: deque[DashboardEvent] = field(default_factory=lambda: deque(maxlen=2_000))
    next_sequence: int = 1


class TestSessionManager:
    """Own worker subprocesses and expose credential-free runtime snapshots."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._sessions: dict[str, _MutableSession] = {}
        self._active_session_id: str | None = None

    def start(
        self,
        *,
        camera_id: int,
        stream_path: str,
        rtsp_url: str,
        sample_fps: float,
        max_samples: int,
        save_snapshots: bool,
        match_faces: bool,
    ) -> TestSessionSnapshot:
        """Start a bounded worker without placing RTSP credentials on its command line."""
        with self._lock:
            if self._active_session_id is not None:
                active = self._sessions[self._active_session_id]
                if active.status not in TERMINAL_SESSION_STATUSES:
                    raise SessionConflictError("another test session is already active")
                self._active_session_id = None

            self._validate_runtime_files(match_faces=match_faces)
            session_id = str(uuid4())
            snapshot_dir = (self.settings.snapshot_dir / f"test-session-{session_id}").resolve()
            session = _MutableSession(
                id=session_id,
                status=TestSessionStatus.STARTING,
                camera_id=camera_id,
                stream_path=stream_path,
                source_name=stream_path,
                rtsp_url=rtsp_url,
                sample_fps=float(sample_fps),
                max_samples=max_samples,
                save_snapshots=save_snapshots,
                match_faces=match_faces,
                snapshot_dir=snapshot_dir,
                started_at=datetime.now(UTC),
                started_monotonic=monotonic(),
            )
            self._sessions[session_id] = session
            self._active_session_id = session_id

            command = self._build_command(session)
            session_log_path = self.settings.log_path.parent / f"test-session-{session_id}.jsonl"
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = self._worker_environment(session, session_log_path)
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                )
            except Exception as error:
                session.status = TestSessionStatus.FAILED
                session.completed_at = datetime.now(UTC)
                session.error_type = type(error).__name__
                self._active_session_id = None
                self._append_event_locked(
                    session,
                    "session_failed",
                    {"error_type": session.error_type},
                )
                raise
            session.process = process
            self._append_event_locked(
                session,
                "session_started",
                {
                    "status": session.status,
                    "camera_id": session.camera_id,
                    "stream_path": session.stream_path,
                    "source_name": session.source_name,
                    "sample_fps": session.sample_fps,
                    "max_samples": session.max_samples,
                },
            )

        Thread(
            target=self._read_worker_output,
            args=(session_id,),
            name=f"test-session-output-{session_id[:8]}",
            daemon=True,
        ).start()
        Thread(
            target=self._monitor_worker,
            args=(session_id,),
            name=f"test-session-monitor-{session_id[:8]}",
            daemon=True,
        ).start()
        return self.get(session_id)

    def stop(self, session_id: str) -> TestSessionSnapshot:
        """Request cooperative SIGTERM shutdown for exactly one owned worker."""
        with self._lock:
            session = self._session_locked(session_id)
            if session.status in TERMINAL_SESSION_STATUSES:
                return self._snapshot_locked(session)
            session.status = TestSessionStatus.STOPPING
            process = session.process
            self._append_event_locked(session, "session_stop_requested", {"status": session.status})
        if process is not None and process.poll() is None:
            process.terminate()
        return self.get(session_id)

    def get(self, session_id: str) -> TestSessionSnapshot:
        with self._lock:
            return self._snapshot_locked(self._session_locked(session_id))

    def list(self) -> tuple[TestSessionSnapshot, ...]:
        with self._lock:
            sessions = sorted(
                self._sessions.values(),
                key=lambda item: item.started_at,
                reverse=True,
            )
            return tuple(self._snapshot_locked(session) for session in sessions)

    def delete(self, session_id: str) -> None:
        """Delete one terminal dashboard session and its session-owned files."""
        with self._lock:
            session = self._session_locked(session_id)
            if session.status not in TERMINAL_SESSION_STATUSES:
                raise SessionConflictError("an active test session cannot be deleted")
            snapshot_dir = session.snapshot_dir
            log_path = self.settings.log_path.parent / f"test-session-{session_id}.jsonl"
            del self._sessions[session_id]

        snapshot_root = self.settings.snapshot_dir.expanduser().resolve()
        if snapshot_dir.parent == snapshot_root and snapshot_dir.name == f"test-session-{session_id}":
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        expected_log_parent = self.settings.log_path.parent.expanduser().resolve()
        resolved_log_path = log_path.expanduser().resolve()
        if (
            resolved_log_path.parent == expected_log_parent
            and resolved_log_path.name == f"test-session-{session_id}.jsonl"
        ):
            resolved_log_path.unlink(missing_ok=True)

    def events(self, session_id: str, *, after: int = 0) -> tuple[DashboardEvent, ...]:
        with self._lock:
            session = self._session_locked(session_id)
            return tuple(event for event in session.events if event.sequence > after)

    def snapshots(self, session_id: str) -> tuple[SnapshotInfo, ...]:
        with self._lock:
            session = self._session_locked(session_id)
            directory = session.snapshot_dir
        if not directory.is_dir():
            return ()
        items: list[SnapshotInfo] = []
        for path in sorted(directory.glob("*.jpg"), reverse=True):
            items.append(
                SnapshotInfo(
                    name=path.name,
                    sample_index=sample_index_from_snapshot_name(path.name),
                    size_bytes=path.stat().st_size,
                )
            )
        return tuple(items)

    def snapshot_path(self, session_id: str, name: str) -> Path:
        if Path(name).name != name or not name.casefold().endswith(".jpg"):
            raise FileNotFoundError(name)
        with self._lock:
            directory = self._session_locked(session_id).snapshot_dir
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def shutdown(self) -> None:
        """Stop the active worker when the API process shuts down."""
        with self._lock:
            active_id = self._active_session_id
        if active_id is None:
            return
        try:
            self.stop(active_id)
        except SessionNotFoundError:
            return
        with self._lock:
            process = self._sessions[active_id].process
        if process is None:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _validate_runtime_files(self, *, match_faces: bool) -> None:
        required = [self.settings.model_path]
        if match_faces:
            required.extend(
                [
                    self.settings.face_detection_model_path,
                    self.settings.face_embedding_model_path,
                ]
            )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))

    def _build_command(self, session: _MutableSession) -> list[str]:
        command = [
            sys.executable,
            "-c",
            "from cctv.workers.rtsp import run; run()",
            "--source-name",
            session.source_name,
            "--sample-fps",
            str(session.sample_fps),
            "--max-samples",
            str(session.max_samples),
            "--analyze",
            "--emit-progress",
        ]
        if session.match_faces:
            command.append("--match-faces")
        if session.save_snapshots:
            command.extend(["--snapshot-dir", str(session.snapshot_dir)])
        return command

    def _worker_environment(self, session: _MutableSession, log_path: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CCTV_APP_ENV": self.settings.app_env,
                "CCTV_APP_MODE": "dry_run",
                "CCTV_AI_DEVICE": self.settings.ai_device.value,
                "CCTV_DATABASE_PATH": str(self.settings.database_path),
                "CCTV_MODEL_PATH": str(self.settings.model_path),
                "CCTV_DETECTOR_TYPE": self.settings.detector_type,
                "CCTV_DETECTOR_ENABLED": "true",
                "CCTV_DETECTOR_INPUT_SIZE": str(
                    self.settings.effective_detector_input_size
                ),
                "CCTV_DETECTOR_CONFIDENCE_THRESHOLD": str(
                    self.settings.effective_detector_confidence_threshold
                ),
                "CCTV_DETECTOR_NMS_THRESHOLD": str(
                    self.settings.effective_detector_nms_threshold
                ),
                "CCTV_DETECTION_CLASS_NAMES": self.settings.detection_class_names,
                "CCTV_TRACKING_ENABLED": str(self.settings.tracking_enabled).lower(),
                "CCTV_TRACKER_CLASS_NAMES": self.settings.tracker_class_names,
                "CCTV_PERSIST_DETECTIONS": "true",
                "CCTV_RULES_ENABLED": str(self.settings.rules_enabled).lower(),
                "CCTV_HIPERWALL_DRY_RUN_ENABLED": str(
                    self.settings.hiperwall_dry_run_enabled
                ).lower(),
                "CCTV_FACE_DETECTION_MODEL_PATH": str(
                    self.settings.face_detection_model_path
                ),
                "CCTV_FACE_EMBEDDING_MODEL_PATH": str(
                    self.settings.face_embedding_model_path
                ),
                "CCTV_FACE_MATCH_SIMILARITY_THRESHOLD": str(
                    self.settings.face_match_similarity_threshold
                ),
                "CCTV_FACE_MATCH_MINIMUM_MARGIN": str(
                    self.settings.face_match_minimum_margin
                ),
                "CCTV_FACE_MATCH_UNKNOWN_RETRY_SECONDS": str(
                    self.settings.face_match_unknown_retry_seconds
                ),
                "CCTV_RTSP_WORKER_URL": session.rtsp_url,
                "CCTV_RTSP_SOURCE_NAME": session.stream_path,
                "CCTV_LOG_LEVEL": self.settings.log_level,
                "CCTV_LOG_PATH": str(log_path),
            }
        )
        if self.settings.model_classes_path is not None:
            environment["CCTV_MODEL_CLASSES_PATH"] = str(self.settings.model_classes_path)
        return environment

    def _read_worker_output(self, session_id: str) -> None:
        with self._lock:
            session = self._session_locked(session_id)
            process = session.process
        if process is None or process.stdout is None:
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
                self._consume_worker_record(session_id, record)

    def _consume_worker_record(self, session_id: str, record: dict[str, Any]) -> None:
        event_name = record.get("event")
        with self._lock:
            session = self._session_locked(session_id)
            if event_name == "analysis_run_created":
                session.analysis_run_id = _optional_string(record.get("analysis_run_id"))
            elif event_name == "rtsp_worker_started":
                session.status = TestSessionStatus.RUNNING
                session.last_message = "RTSP 워커가 시작되었습니다."
            elif event_name == "rtsp_source_connected":
                session.connection_count = int(record.get("connection_count", 0))
                session.last_message = "RTSP 스트림에 연결되었습니다."
            elif event_name == "rtsp_worker_progress":
                session.processed_samples = int(record.get("processed_samples", 0))
            elif event_name == "video_face_similarity_evaluated":
                session.last_message = _face_event_message(record)
            elif event_name == "rtsp_worker_finished":
                session.processed_samples = int(record.get("processed_samples", 0))
                session.connection_count = int(record.get("connection_count", 0))
                session.reconnect_count = int(record.get("reconnect_count", 0))
                session.elapsed_seconds = float(record.get("elapsed_seconds", 0))
            elif event_name == "analysis_run_finished":
                session.analysis_run_id = _optional_string(record.get("analysis_run_id"))
                session.error_type = _optional_string(record.get("error_type"))
            elif event_name == "rtsp_worker_failed":
                session.error_type = _optional_string(record.get("error_type")) or "WorkerError"
                session.last_message = "RTSP 워커 실행에 실패했습니다."
            elif not event_name and "processed_samples" in record and "status" in record:
                self._apply_final_summary_locked(session, record)

            if isinstance(event_name, str) and event_name in _FORWARDED_WORKER_EVENTS:
                self._append_event_locked(
                    session,
                    event_name,
                    _public_worker_event(record),
                )

    def _monitor_worker(self, session_id: str) -> None:
        with self._lock:
            session = self._session_locked(session_id)
            process = session.process
        if process is None:
            return
        metrics = _ProcMetricsSampler(process.pid)
        while process.poll() is None:
            sleep(0.5)
            sample = metrics.sample()
            with self._lock:
                session = self._session_locked(session_id)
                session.elapsed_seconds = round(monotonic() - session.started_monotonic, 3)
                if sample is not None:
                    session.cpu_percent, session.memory_mib = sample
                    session.peak_cpu_percent = _maximum(
                        session.peak_cpu_percent,
                        session.cpu_percent,
                    )
                    session.peak_memory_mib = _maximum(
                        session.peak_memory_mib,
                        session.memory_mib,
                    )
                self._refresh_analysis_progress_locked(session)
                self._append_event_locked(
                    session,
                    "session_progress",
                    {
                        "status": session.status,
                        "processed_samples": session.processed_samples,
                        "max_samples": session.max_samples,
                        "elapsed_seconds": session.elapsed_seconds,
                        "cpu_percent": session.cpu_percent,
                        "memory_mib": session.memory_mib,
                    },
                )

        process.wait()
        with self._lock:
            session = self._session_locked(session_id)
            session.elapsed_seconds = round(monotonic() - session.started_monotonic, 3)
            self._refresh_analysis_progress_locked(session)
            if session.result_summary is not None:
                result_status = session.result_summary.get("status")
                session.status = (
                    TestSessionStatus.COMPLETED
                    if result_status == "completed"
                    else TestSessionStatus.STOPPED
                )
            elif session.status is TestSessionStatus.STOPPING:
                session.status = TestSessionStatus.STOPPED
            elif process.returncode == 0:
                session.status = TestSessionStatus.COMPLETED
            else:
                session.status = TestSessionStatus.FAILED
                session.error_type = session.error_type or "WorkerProcessError"
            session.completed_at = datetime.now(UTC)
            session.cpu_percent = None
            session.memory_mib = None
            if self._active_session_id == session_id:
                self._active_session_id = None
            self._append_event_locked(
                session,
                "session_finished",
                {
                    "status": session.status,
                    "processed_samples": session.processed_samples,
                    "elapsed_seconds": session.elapsed_seconds,
                    "error_type": session.error_type,
                },
            )

    def _refresh_analysis_progress_locked(self, session: _MutableSession) -> None:
        if session.analysis_run_id is None:
            return
        try:
            run = DetectionRepository(self.settings.database_path).get_analysis_run(
                session.analysis_run_id
            )
        except (OSError, sqlite3.Error):
            return
        if run is not None:
            session.processed_samples = max(session.processed_samples, run.processed_frames)

    def _apply_final_summary_locked(
        self,
        session: _MutableSession,
        record: dict[str, Any],
    ) -> None:
        session.result_summary = _public_final_summary(record)
        session.processed_samples = int(record.get("processed_samples", 0))
        session.connection_count = int(record.get("connection_count", 0))
        session.reconnect_count = int(record.get("reconnect_count", 0))
        session.elapsed_seconds = float(record.get("elapsed_seconds", 0))
        analysis = record.get("analysis")
        if isinstance(analysis, dict):
            session.analysis_run_id = _optional_string(analysis.get("analysis_run_id"))

    def _session_locked(self, session_id: str) -> _MutableSession:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(session_id) from error

    def _snapshot_locked(self, session: _MutableSession) -> TestSessionSnapshot:
        snapshot_count = (
            sum(1 for _ in session.snapshot_dir.glob("*.jpg"))
            if session.snapshot_dir.is_dir()
            else 0
        )
        return TestSessionSnapshot(
            id=session.id,
            status=session.status,
            camera_id=session.camera_id,
            stream_path=session.stream_path,
            source_name=session.source_name,
            sample_fps=session.sample_fps,
            max_samples=session.max_samples,
            save_snapshots=session.save_snapshots,
            match_faces=session.match_faces,
            analysis_run_id=session.analysis_run_id,
            processed_samples=session.processed_samples,
            connection_count=session.connection_count,
            reconnect_count=session.reconnect_count,
            elapsed_seconds=session.elapsed_seconds,
            cpu_percent=session.cpu_percent,
            memory_mib=session.memory_mib,
            peak_cpu_percent=session.peak_cpu_percent,
            peak_memory_mib=session.peak_memory_mib,
            snapshot_count=snapshot_count,
            started_at=session.started_at,
            completed_at=session.completed_at,
            last_message=session.last_message,
            error_type=session.error_type,
            result_summary=session.result_summary,
        )

    def _append_event_locked(
        self,
        session: _MutableSession,
        event: str,
        data: dict[str, Any],
    ) -> None:
        session.events.append(
            DashboardEvent(
                sequence=session.next_sequence,
                event=event,
                timestamp=datetime.now(UTC),
                data=data,
            )
        )
        session.next_sequence += 1


class _ProcMetricsSampler:
    """Read Linux process CPU and RSS without adding a runtime dependency."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._clock_ticks = _system_configuration("SC_CLK_TCK", 100.0)
        self._page_size = _system_configuration("SC_PAGE_SIZE", 4096.0)
        self._previous_cpu_seconds: float | None = None
        self._previous_monotonic: float | None = None

    def sample(self) -> tuple[float, float] | None:
        try:
            stat_fields = Path(f"/proc/{self.pid}/stat").read_text(encoding="utf-8").split()
            resident_pages = int(
                Path(f"/proc/{self.pid}/statm").read_text(encoding="utf-8").split()[1]
            )
            cpu_seconds = (int(stat_fields[13]) + int(stat_fields[14])) / self._clock_ticks
        except (FileNotFoundError, IndexError, OSError, ValueError):
            return None
        if resident_pages <= 0:
            return None
        sampled_at = monotonic()
        cpu_percent = 0.0
        if self._previous_cpu_seconds is not None and self._previous_monotonic is not None:
            elapsed = sampled_at - self._previous_monotonic
            if elapsed > 0:
                cpu_percent = max(
                    0.0,
                    (cpu_seconds - self._previous_cpu_seconds) / elapsed * 100,
                )
        self._previous_cpu_seconds = cpu_seconds
        self._previous_monotonic = sampled_at
        memory_mib = resident_pages * self._page_size / (1024 * 1024)
        return round(cpu_percent, 2), round(memory_mib, 2)


_FORWARDED_WORKER_EVENTS = {
    "analysis_run_created",
    "face_matching_pipeline_initialized",
    "rtsp_worker_started",
    "rtsp_source_connected",
    "rtsp_source_reconnecting",
    "video_face_similarity_evaluated",
    "rule_event_emitted",
    "face_matching_run_completed",
    "rtsp_worker_finished",
    "rtsp_worker_failed",
}

_PUBLIC_EVENT_FIELDS = {
    "analysis_run_id",
    "source_name",
    "sample_fps",
    "max_samples",
    "connection_count",
    "reconnect_count",
    "processed_samples",
    "elapsed_seconds",
    "candidate_identity_count",
    "candidate_embedding_count",
    "similarity_threshold",
    "minimum_margin",
    "source_index",
    "sample_index",
    "timestamp_seconds",
    "face_index",
    "track_id",
    "rule_id",
    "event_type",
    "event_state",
    "class_name",
    "confidence",
    "payload",
    "face_bounds",
    "detection_confidence",
    "match_status",
    "rejection_reason",
    "identity_id",
    "external_id",
    "display_name",
    "best_similarity",
    "second_best_similarity",
    "frames_with_faces",
    "detected_faces",
    "matched_faces",
    "unknown_faces",
    "face_analysis_attempts",
    "cache_hits",
    "cached_tracks",
    "error_type",
}


def _public_worker_event(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in _PUBLIC_EVENT_FIELDS if key in record}


def _public_final_summary(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: record.get(key)
        for key in (
            "status",
            "stop_reason",
            "source_name",
            "sample_fps",
            "processed_samples",
            "elapsed_seconds",
            "connection_count",
            "reconnect_count",
            "decoded_frames",
        )
    }
    analysis = record.get("analysis")
    if isinstance(analysis, dict):
        result["analysis"] = {
            key: analysis.get(key)
            for key in (
                "analysis_run_id",
                "detector_type",
                "device",
                "processed_frames",
                "total_detections",
                "average_inference_seconds",
                "tracking_enabled",
                "tracking",
                "upper_body_color",
                "rules",
                "hiperwall",
            )
        }
    face_matching = record.get("face_matching")
    if isinstance(face_matching, dict):
        result["face_matching"] = face_matching
    snapshots = record.get("snapshots")
    if isinstance(snapshots, dict):
        result["snapshots"] = {
            "saved_count": snapshots.get("saved_count"),
            "jpeg_quality": snapshots.get("jpeg_quality"),
        }
    return result


def _face_event_message(record: dict[str, Any]) -> str:
    status = str(record.get("match_status", "unknown"))
    if record.get("rejection_reason") == "no_candidates":
        return f"얼굴 판정: {status} · 등록 후보 없음"
    similarity = record.get("best_similarity")
    label = record.get("external_id") or record.get("display_name") or "unknown"
    if isinstance(similarity, (int, float)):
        return f"얼굴 판정: {status} · {label} · {float(similarity):.3f}"
    return f"얼굴 판정: {status} · {label}"


def _optional_string(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _maximum(previous: float | None, current: float | None) -> float | None:
    if current is None:
        return previous
    return current if previous is None else max(previous, current)


def _system_configuration(name: str, fallback: float) -> float:
    try:
        return float(os.sysconf(name))
    except (AttributeError, OSError, ValueError):
        return fallback


def event_to_dict(event: DashboardEvent) -> dict[str, Any]:
    """Convert a dashboard event into a JSON-compatible object."""
    return asdict(event)


__all__ = [
    "TERMINAL_SESSION_STATUSES",
    "DashboardEvent",
    "SessionConflictError",
    "SessionNotFoundError",
    "SnapshotInfo",
    "TestSessionManager",
    "TestSessionSnapshot",
    "TestSessionStatus",
    "event_to_dict",
]
