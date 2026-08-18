"""SQLite repository for analysis runs, analyzed frames, and detections."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from cctv.db.database import connect_database

if TYPE_CHECKING:
    from cctv.inference import FrameDetections

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100

logger = logging.getLogger(__name__)


class AnalysisRunStatus(StrEnum):
    """Persistent lifecycle state for one video analysis run."""

    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AnalysisRunRecord:
    """One persisted analysis run without internal filesystem paths."""

    id: str
    source_type: str
    source_name: str
    camera_id: int | None
    model_name: str
    model_sha256: str
    device: str
    input_size: int
    confidence_threshold: float
    nms_threshold: float
    sample_fps: float
    status: AnalysisRunStatus
    started_at: datetime
    completed_at: datetime | None
    processed_frames: int
    total_detections: int
    error_type: str | None


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """One persisted detection joined with its source-frame metadata."""

    id: int
    analysis_run_id: str
    analyzed_frame_id: int
    source_index: int
    sample_index: int
    source_timestamp_seconds: float
    frame_width: int
    frame_height: int
    inference_seconds: float
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True, slots=True)
class AnalysisRunPage:
    """One bounded page of analysis runs."""

    items: tuple[AnalysisRunRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class DetectionPage:
    """One bounded page of detection rows."""

    items: tuple[DetectionRecord, ...]
    total: int


class DetectionRepository:
    """Persist and query normalized detection results using short transactions."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def create_analysis_run(
        self,
        *,
        source_type: str,
        source_name: str,
        model_name: str,
        model_sha256: str,
        device: str,
        input_size: int,
        confidence_threshold: float,
        nms_threshold: float,
        sample_fps: float,
        camera_id: int | None = None,
        run_id: str | None = None,
    ) -> AnalysisRunRecord:
        """Create a running analysis record before frames are processed."""
        normalized_source_name = source_name.strip()
        normalized_model_name = model_name.strip()
        normalized_sha256 = model_sha256.strip().lower()
        if source_type not in {"local_video", "rtsp"}:
            raise ValueError("source_type must be local_video or rtsp")
        if not normalized_source_name:
            raise ValueError("source_name must not be empty")
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")
        if len(normalized_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_sha256
        ):
            raise ValueError("model_sha256 must be a 64-character hexadecimal digest")

        identifier = run_id or str(uuid4())
        started_at = _utc_now()
        with closing(connect_database(self.database_path)) as connection, connection:
            connection.execute(
                """
                    INSERT INTO analysis_runs (
                        id, source_type, source_name, camera_id, model_name, model_sha256,
                        device, input_size, confidence_threshold, nms_threshold, sample_fps,
                        status, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    identifier,
                    source_type,
                    normalized_source_name,
                    camera_id,
                    normalized_model_name,
                    normalized_sha256,
                    device,
                    input_size,
                    confidence_threshold,
                    nms_threshold,
                    sample_fps,
                    AnalysisRunStatus.RUNNING,
                    _datetime_to_text(started_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise RuntimeError("analysis run was not created")
        record = _analysis_run_from_row(row)
        logger.info(
            "Analysis run created",
            extra={
                "event": "analysis_run_created",
                "analysis_run_id": record.id,
                "source_type": record.source_type,
                "source_name": record.source_name,
                "model_name": record.model_name,
                "model_sha256": record.model_sha256,
            },
        )
        return record

    def save_frame(self, analysis_run_id: str, result: FrameDetections) -> int:
        """Atomically persist one analyzed frame and every detection it contains."""
        with closing(connect_database(self.database_path)) as connection, connection:
            run = connection.execute(
                "SELECT status FROM analysis_runs WHERE id = ?",
                (analysis_run_id,),
            ).fetchone()
            if run is None:
                raise LookupError(f"analysis run does not exist: {analysis_run_id}")
            if run["status"] != AnalysisRunStatus.RUNNING:
                raise RuntimeError("detections can only be added to a running analysis")

            cursor = connection.execute(
                """
                    INSERT INTO analyzed_frames (
                        analysis_run_id, source_index, sample_index,
                        source_timestamp_seconds, frame_width, frame_height,
                        inference_seconds, detection_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    analysis_run_id,
                    result.source_index,
                    result.sample_index,
                    result.timestamp_seconds,
                    result.frame_width,
                    result.frame_height,
                    result.inference_seconds,
                    len(result.detections),
                ),
            )
            frame_id = cursor.lastrowid
            if frame_id is None:
                raise RuntimeError("analyzed frame did not receive an identifier")
            connection.executemany(
                """
                    INSERT INTO detections (
                        analyzed_frame_id, class_id, class_name, confidence,
                        x1, y1, x2, y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    (
                        frame_id,
                        detection.class_id,
                        detection.label,
                        detection.confidence,
                        detection.box.x1,
                        detection.box.y1,
                        detection.box.x2,
                        detection.box.y2,
                    )
                    for detection in result.detections
                ),
            )
        persisted_frame_id = int(frame_id)
        logger.debug(
            "Frame detections persisted",
            extra={
                "event": "frame_detections_persisted",
                "analysis_run_id": analysis_run_id,
                "analyzed_frame_id": persisted_frame_id,
                "sample_index": result.sample_index,
                "detection_count": len(result.detections),
            },
        )
        return persisted_frame_id

    def finish_analysis_run(
        self,
        analysis_run_id: str,
        *,
        status: AnalysisRunStatus,
        error_type: str | None = None,
    ) -> AnalysisRunRecord:
        """Finish one running analysis and derive persisted aggregate counts."""
        if status is AnalysisRunStatus.RUNNING:
            raise ValueError("a finished analysis run cannot remain running")
        completed_at = _utc_now()

        with closing(connect_database(self.database_path)) as connection, connection:
            counts = connection.execute(
                """
                    SELECT
                        COUNT(DISTINCT frame.id) AS processed_frames,
                        COUNT(detection.id) AS total_detections
                    FROM analyzed_frames AS frame
                    LEFT JOIN detections AS detection
                        ON detection.analyzed_frame_id = frame.id
                    WHERE frame.analysis_run_id = ?
                    """,
                (analysis_run_id,),
            ).fetchone()
            cursor = connection.execute(
                """
                    UPDATE analysis_runs
                    SET status = ?, completed_at = ?, processed_frames = ?,
                        total_detections = ?, error_type = ?
                    WHERE id = ? AND status = 'running'
                    """,
                (
                    status,
                    _datetime_to_text(completed_at),
                    int(counts["processed_frames"]),
                    int(counts["total_detections"]),
                    error_type if status is AnalysisRunStatus.FAILED else None,
                    analysis_run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"running analysis run does not exist: {analysis_run_id}")
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE id = ?",
                (analysis_run_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("finished analysis run could not be read")
        record = _analysis_run_from_row(row)
        logger.info(
            "Analysis run finished",
            extra={
                "event": "analysis_run_finished",
                "analysis_run_id": record.id,
                "status": record.status,
                "processed_frames": record.processed_frames,
                "total_detections": record.total_detections,
                "error_type": record.error_type,
            },
        )
        return record

    def get_analysis_run(self, analysis_run_id: str) -> AnalysisRunRecord | None:
        """Return one analysis run by public identifier."""
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE id = ?",
                (analysis_run_id,),
            ).fetchone()
        return _analysis_run_from_row(row) if row is not None else None

    def list_analysis_runs(
        self,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        status: AnalysisRunStatus | None = None,
    ) -> AnalysisRunPage:
        """List recent analysis runs with stable, bounded offset pagination."""
        _validate_pagination(page, limit)
        where_sql = " WHERE status = ?" if status is not None else ""
        parameters: tuple[object, ...] = (status,) if status is not None else ()
        offset = (page - 1) * limit

        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM analysis_runs{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM analysis_runs{where_sql}
                ORDER BY started_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return AnalysisRunPage(
            items=tuple(_analysis_run_from_row(row) for row in rows),
            total=total,
        )

    def list_detections(
        self,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        analysis_run_id: str | None = None,
        class_name: str | None = None,
        min_confidence: float | None = None,
    ) -> DetectionPage:
        """Query detections by run, class, and minimum confidence."""
        _validate_pagination(page, limit)
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        clauses: list[str] = []
        parameters: list[object] = []
        if analysis_run_id is not None:
            clauses.append("frame.analysis_run_id = ?")
            parameters.append(analysis_run_id)
        if class_name is not None:
            normalized_class_name = class_name.strip()
            if not normalized_class_name:
                raise ValueError("class_name must not be empty")
            clauses.append("detection.class_name = ? COLLATE NOCASE")
            parameters.append(normalized_class_name)
        if min_confidence is not None:
            clauses.append("detection.confidence >= ?")
            parameters.append(min_confidence)

        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        join_sql = """
            FROM detections AS detection
            JOIN analyzed_frames AS frame ON frame.id = detection.analyzed_frame_id
            JOIN analysis_runs AS run ON run.id = frame.analysis_run_id
        """
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) {join_sql}{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT
                    detection.id,
                    frame.analysis_run_id,
                    frame.id AS analyzed_frame_id,
                    frame.source_index,
                    frame.sample_index,
                    frame.source_timestamp_seconds,
                    frame.frame_width,
                    frame.frame_height,
                    frame.inference_seconds,
                    detection.class_id,
                    detection.class_name,
                    detection.confidence,
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2
                {join_sql}{where_sql}
                ORDER BY run.started_at DESC, frame.sample_index ASC,
                    detection.confidence DESC, detection.id ASC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return DetectionPage(
            items=tuple(_detection_from_row(row) for row in rows),
            total=total,
        )


def _validate_pagination(page: int, limit: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _datetime_to_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _datetime_from_text(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _analysis_run_from_row(row: sqlite3.Row) -> AnalysisRunRecord:
    started_at = _datetime_from_text(row["started_at"])
    if started_at is None:
        raise RuntimeError("analysis run is missing started_at")
    return AnalysisRunRecord(
        id=str(row["id"]),
        source_type=str(row["source_type"]),
        source_name=str(row["source_name"]),
        camera_id=int(row["camera_id"]) if row["camera_id"] is not None else None,
        model_name=str(row["model_name"]),
        model_sha256=str(row["model_sha256"]),
        device=str(row["device"]),
        input_size=int(row["input_size"]),
        confidence_threshold=float(row["confidence_threshold"]),
        nms_threshold=float(row["nms_threshold"]),
        sample_fps=float(row["sample_fps"]),
        status=AnalysisRunStatus(row["status"]),
        started_at=started_at,
        completed_at=_datetime_from_text(row["completed_at"]),
        processed_frames=int(row["processed_frames"]),
        total_detections=int(row["total_detections"]),
        error_type=str(row["error_type"]) if row["error_type"] is not None else None,
    )


def _detection_from_row(row: sqlite3.Row) -> DetectionRecord:
    return DetectionRecord(
        id=int(row["id"]),
        analysis_run_id=str(row["analysis_run_id"]),
        analyzed_frame_id=int(row["analyzed_frame_id"]),
        source_index=int(row["source_index"]),
        sample_index=int(row["sample_index"]),
        source_timestamp_seconds=float(row["source_timestamp_seconds"]),
        frame_width=int(row["frame_width"]),
        frame_height=int(row["frame_height"]),
        inference_seconds=float(row["inference_seconds"]),
        class_id=int(row["class_id"]),
        class_name=str(row["class_name"]),
        confidence=float(row["confidence"]),
        x1=int(row["x1"]),
        y1=int(row["y1"]),
        x2=int(row["x2"]),
        y2=int(row["y2"]),
    )
