"""SQLite persistence for privacy-safe face recognition events."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cctv.db.database import connect_database
from cctv.db.detections import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE


@dataclass(frozen=True, slots=True)
class FaceMatchEventInput:
    """One fresh face comparison ready for persistence."""

    source_index: int
    sample_index: int
    source_timestamp_seconds: float
    face_index: int
    track_id: int | None
    face_x: int
    face_y: int
    face_width: int
    face_height: int
    detection_confidence: float
    match_status: str
    rejection_reason: str | None
    identity_id: str | None
    external_id: str | None
    display_name: str | None
    best_candidate_identity_id: str
    best_candidate_external_id: str | None
    best_similarity: float
    second_best_similarity: float | None
    similarity_threshold: float
    minimum_margin: float


@dataclass(frozen=True, slots=True)
class FaceMatchEventRecord:
    """Stored face comparison without an embedding vector or source credentials."""

    id: int
    analysis_run_id: str
    source_index: int
    sample_index: int
    source_timestamp_seconds: float
    face_index: int
    track_id: int | None
    face_x: int
    face_y: int
    face_width: int
    face_height: int
    detection_confidence: float
    match_status: str
    rejection_reason: str | None
    identity_id: str | None
    external_id: str | None
    display_name: str | None
    best_candidate_identity_id: str
    best_candidate_external_id: str | None
    best_similarity: float
    second_best_similarity: float | None
    similarity_threshold: float
    minimum_margin: float
    created_at: datetime


@dataclass(frozen=True, slots=True)
class FaceMatchEventPage:
    items: tuple[FaceMatchEventRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class FaceMatchRunSummary:
    analysis_run_id: str
    detected_faces: int
    matched_faces: int
    unknown_faces: int
    best_similarity: float | None


class FaceMatchRepository:
    """Store and query fresh face-comparison decisions by analysis run."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def save_event(
        self,
        analysis_run_id: str,
        event: FaceMatchEventInput,
    ) -> FaceMatchEventRecord:
        with closing(connect_database(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO face_match_events (
                    analysis_run_id, source_index, sample_index,
                    source_timestamp_seconds, face_index, track_id,
                    face_x, face_y, face_width, face_height,
                    detection_confidence, match_status, rejection_reason,
                    identity_id, external_id, display_name,
                    best_candidate_identity_id, best_candidate_external_id,
                    best_similarity, second_best_similarity,
                    similarity_threshold, minimum_margin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_run_id,
                    event.source_index,
                    event.sample_index,
                    event.source_timestamp_seconds,
                    event.face_index,
                    event.track_id,
                    event.face_x,
                    event.face_y,
                    event.face_width,
                    event.face_height,
                    event.detection_confidence,
                    event.match_status,
                    event.rejection_reason,
                    event.identity_id,
                    event.external_id,
                    event.display_name,
                    event.best_candidate_identity_id,
                    event.best_candidate_external_id,
                    event.best_similarity,
                    event.second_best_similarity,
                    event.similarity_threshold,
                    event.minimum_margin,
                ),
            )
            row = connection.execute(
                "SELECT * FROM face_match_events WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        if row is None:
            raise RuntimeError("face match event was not created")
        return _event_from_row(row)

    def list_events(
        self,
        *,
        analysis_run_id: str,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        match_status: str | None = None,
    ) -> FaceMatchEventPage:
        _validate_pagination(page, limit)
        clauses = ["analysis_run_id = ?"]
        parameters: list[object] = [analysis_run_id]
        if match_status is not None:
            if match_status not in {"matched", "unknown"}:
                raise ValueError("match_status must be matched or unknown")
            clauses.append("match_status = ?")
            parameters.append(match_status)
        where_sql = " WHERE " + " AND ".join(clauses)
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM face_match_events{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM face_match_events{where_sql}
                ORDER BY sample_index DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return FaceMatchEventPage(
            items=tuple(_event_from_row(row) for row in rows),
            total=total,
        )

    def summarize_run(self, analysis_run_id: str) -> FaceMatchRunSummary:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS detected_faces,
                    SUM(CASE WHEN match_status = 'matched' THEN 1 ELSE 0 END)
                        AS matched_faces,
                    SUM(CASE WHEN match_status = 'unknown' THEN 1 ELSE 0 END)
                        AS unknown_faces,
                    MAX(best_similarity) AS best_similarity
                FROM face_match_events
                WHERE analysis_run_id = ?
                """,
                (analysis_run_id,),
            ).fetchone()
        return FaceMatchRunSummary(
            analysis_run_id=analysis_run_id,
            detected_faces=int(row["detected_faces"]),
            matched_faces=int(row["matched_faces"] or 0),
            unknown_faces=int(row["unknown_faces"] or 0),
            best_similarity=(
                float(row["best_similarity"]) if row["best_similarity"] is not None else None
            ),
        )


def _event_from_row(row: sqlite3.Row) -> FaceMatchEventRecord:
    created_at = datetime.fromisoformat(str(row["created_at"]))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return FaceMatchEventRecord(
        id=int(row["id"]),
        analysis_run_id=str(row["analysis_run_id"]),
        source_index=int(row["source_index"]),
        sample_index=int(row["sample_index"]),
        source_timestamp_seconds=float(row["source_timestamp_seconds"]),
        face_index=int(row["face_index"]),
        track_id=int(row["track_id"]) if row["track_id"] is not None else None,
        face_x=int(row["face_x"]),
        face_y=int(row["face_y"]),
        face_width=int(row["face_width"]),
        face_height=int(row["face_height"]),
        detection_confidence=float(row["detection_confidence"]),
        match_status=str(row["match_status"]),
        rejection_reason=(
            str(row["rejection_reason"]) if row["rejection_reason"] is not None else None
        ),
        identity_id=str(row["identity_id"]) if row["identity_id"] is not None else None,
        external_id=str(row["external_id"]) if row["external_id"] is not None else None,
        display_name=str(row["display_name"]) if row["display_name"] is not None else None,
        best_candidate_identity_id=str(row["best_candidate_identity_id"]),
        best_candidate_external_id=(
            str(row["best_candidate_external_id"])
            if row["best_candidate_external_id"] is not None
            else None
        ),
        best_similarity=float(row["best_similarity"]),
        second_best_similarity=(
            float(row["second_best_similarity"])
            if row["second_best_similarity"] is not None
            else None
        ),
        similarity_threshold=float(row["similarity_threshold"]),
        minimum_margin=float(row["minimum_margin"]),
        created_at=created_at,
    )


def _validate_pagination(page: int, limit: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


__all__ = [
    "FaceMatchEventInput",
    "FaceMatchEventPage",
    "FaceMatchEventRecord",
    "FaceMatchRepository",
    "FaceMatchRunSummary",
]

