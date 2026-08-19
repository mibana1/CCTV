"""SQLite persistence for session-scoped people and their fragmented tracks."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from cctv.db.database import connect_database
from cctv.db.detections import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from cctv.db.face_matches import FaceMatchEventRecord


@dataclass(frozen=True, slots=True)
class PersonInstanceRecord:
    """One session-scoped person composed of one or more tracker IDs."""

    id: str
    analysis_run_id: str
    status: str
    identity_id: str | None
    external_id: str | None
    display_name: str | None
    best_candidate_identity_id: str
    best_candidate_external_id: str | None
    best_similarity: float
    first_sample_index: int
    last_sample_index: int
    first_seen_timestamp_seconds: float
    last_seen_timestamp_seconds: float
    last_face_x: int
    last_face_y: int
    last_face_width: int
    last_face_height: int
    track_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TrackIdentityLinkRecord:
    """Association between an execution-scoped object track and a person instance."""

    id: int
    analysis_run_id: str
    track_id: int
    person_instance_id: str
    linked_by: str
    confidence: float
    first_sample_index: int
    last_sample_index: int
    first_seen_timestamp_seconds: float
    last_seen_timestamp_seconds: float
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PersonInstancePage:
    items: tuple[PersonInstanceRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class TrackIdentityLinkPage:
    items: tuple[TrackIdentityLinkRecord, ...]
    total: int


@dataclass(frozen=True, slots=True)
class PersonInstanceRunSummary:
    analysis_run_id: str
    total_instances: int
    matched_instances: int
    unknown_instances: int
    linked_tracks: int


class PersonInstanceRepository:
    """Create, merge, and query stable people inferred from short-lived tracks."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def create_from_event(
        self,
        event: FaceMatchEventRecord,
        *,
        linked_by: str,
        confidence: float,
        instance_id: str | None = None,
    ) -> PersonInstanceRecord:
        track_id = _required_track_id(event)
        _validate_link(linked_by, confidence)
        identifier = instance_id or str(uuid4())
        status = "matched" if event.match_status == "matched" else "unknown"
        with closing(connect_database(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO person_instances (
                    id, analysis_run_id, status, identity_id, external_id, display_name,
                    best_candidate_identity_id, best_candidate_external_id, best_similarity,
                    first_sample_index, last_sample_index,
                    first_seen_timestamp_seconds, last_seen_timestamp_seconds,
                    last_face_x, last_face_y, last_face_width, last_face_height,
                    track_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    identifier,
                    event.analysis_run_id,
                    status,
                    event.identity_id,
                    event.external_id,
                    event.display_name,
                    event.best_candidate_identity_id,
                    event.best_candidate_external_id,
                    event.best_similarity,
                    event.sample_index,
                    event.sample_index,
                    event.source_timestamp_seconds,
                    event.source_timestamp_seconds,
                    event.face_x,
                    event.face_y,
                    event.face_width,
                    event.face_height,
                ),
            )
            _upsert_link(connection, identifier, event, track_id, linked_by, confidence)
            _assign_event(connection, event, identifier)
            row = connection.execute(
                "SELECT * FROM person_instances WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            raise RuntimeError("person instance was not created")
        return _person_instance_from_row(row)

    def apply_event(
        self,
        instance_id: str,
        event: FaceMatchEventRecord,
        *,
        linked_by: str,
        confidence: float,
    ) -> PersonInstanceRecord:
        track_id = _required_track_id(event)
        _validate_link(linked_by, confidence)
        with closing(connect_database(self.database_path)) as connection, connection:
            row = connection.execute(
                "SELECT * FROM person_instances WHERE id = ?",
                (instance_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"person instance does not exist: {instance_id}")
            current = _person_instance_from_row(row)
            if current.analysis_run_id != event.analysis_run_id:
                raise ValueError("face event and person instance must belong to the same run")
            if (
                current.identity_id is not None
                and event.identity_id is not None
                and current.identity_id != event.identity_id
            ):
                raise ValueError("one person instance cannot contain different identities")

            _upsert_link(connection, instance_id, event, track_id, linked_by, confidence)
            _assign_event(connection, event, instance_id)
            first_sample = min(current.first_sample_index, event.sample_index)
            last_sample = max(current.last_sample_index, event.sample_index)
            first_seen = min(
                current.first_seen_timestamp_seconds,
                event.source_timestamp_seconds,
            )
            last_seen = max(
                current.last_seen_timestamp_seconds,
                event.source_timestamp_seconds,
            )
            event_is_latest = event.source_timestamp_seconds >= current.last_seen_timestamp_seconds
            event_is_best = event.best_similarity >= current.best_similarity
            identity_id = current.identity_id or event.identity_id
            status = "matched" if identity_id is not None else "unknown"
            track_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM track_identity_links WHERE person_instance_id = ?",
                    (instance_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """
                UPDATE person_instances
                SET status = ?, identity_id = ?, external_id = ?, display_name = ?,
                    best_candidate_identity_id = ?, best_candidate_external_id = ?,
                    best_similarity = ?, first_sample_index = ?, last_sample_index = ?,
                    first_seen_timestamp_seconds = ?, last_seen_timestamp_seconds = ?,
                    last_face_x = ?, last_face_y = ?, last_face_width = ?,
                    last_face_height = ?, track_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status,
                    identity_id,
                    event.external_id or current.external_id,
                    event.display_name or current.display_name,
                    (
                        event.best_candidate_identity_id
                        if event_is_best
                        else current.best_candidate_identity_id
                    ),
                    (
                        event.best_candidate_external_id
                        if event_is_best
                        else current.best_candidate_external_id
                    ),
                    max(current.best_similarity, event.best_similarity),
                    first_sample,
                    last_sample,
                    first_seen,
                    last_seen,
                    event.face_x if event_is_latest else current.last_face_x,
                    event.face_y if event_is_latest else current.last_face_y,
                    event.face_width if event_is_latest else current.last_face_width,
                    event.face_height if event_is_latest else current.last_face_height,
                    track_count,
                    instance_id,
                ),
            )
            updated_row = connection.execute(
                "SELECT * FROM person_instances WHERE id = ?",
                (instance_id,),
            ).fetchone()
        if updated_row is None:
            raise RuntimeError("updated person instance could not be loaded")
        return _person_instance_from_row(updated_row)

    def merge_instances(
        self,
        target_id: str,
        source_id: str,
    ) -> PersonInstanceRecord:
        """Move every source track and event into one compatible target instance."""
        if target_id == source_id:
            result = self.get_instance(target_id)
            if result is None:
                raise LookupError(f"person instance does not exist: {target_id}")
            return result
        with closing(connect_database(self.database_path)) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM person_instances WHERE id IN (?, ?)",
                (target_id, source_id),
            ).fetchall()
            by_id = {str(row["id"]): _person_instance_from_row(row) for row in rows}
            if target_id not in by_id or source_id not in by_id:
                raise LookupError("target and source person instances must both exist")
            target = by_id[target_id]
            source = by_id[source_id]
            if target.analysis_run_id != source.analysis_run_id:
                raise ValueError("person instances from different runs cannot be merged")
            if (
                target.identity_id is not None
                and source.identity_id is not None
                and target.identity_id != source.identity_id
            ):
                raise ValueError("person instances with different identities cannot be merged")

            connection.execute(
                "UPDATE track_identity_links SET person_instance_id = ? WHERE person_instance_id = ?",
                (target_id, source_id),
            )
            connection.execute(
                "UPDATE face_match_events SET person_instance_id = ? WHERE person_instance_id = ?",
                (target_id, source_id),
            )
            identity_id = target.identity_id or source.identity_id
            identified = target if target.identity_id is not None else source
            latest = (
                target
                if target.last_seen_timestamp_seconds >= source.last_seen_timestamp_seconds
                else source
            )
            best = target if target.best_similarity >= source.best_similarity else source
            track_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM track_identity_links WHERE person_instance_id = ?",
                    (target_id,),
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM person_instances WHERE id = ?", (source_id,))
            connection.execute(
                """
                UPDATE person_instances
                SET status = ?, identity_id = ?, external_id = ?, display_name = ?,
                    best_candidate_identity_id = ?, best_candidate_external_id = ?,
                    best_similarity = ?, first_sample_index = ?, last_sample_index = ?,
                    first_seen_timestamp_seconds = ?, last_seen_timestamp_seconds = ?,
                    last_face_x = ?, last_face_y = ?, last_face_width = ?,
                    last_face_height = ?, track_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    "matched" if identity_id is not None else "unknown",
                    identity_id,
                    identified.external_id,
                    identified.display_name,
                    best.best_candidate_identity_id,
                    best.best_candidate_external_id,
                    best.best_similarity,
                    min(target.first_sample_index, source.first_sample_index),
                    max(target.last_sample_index, source.last_sample_index),
                    min(
                        target.first_seen_timestamp_seconds,
                        source.first_seen_timestamp_seconds,
                    ),
                    max(
                        target.last_seen_timestamp_seconds,
                        source.last_seen_timestamp_seconds,
                    ),
                    latest.last_face_x,
                    latest.last_face_y,
                    latest.last_face_width,
                    latest.last_face_height,
                    track_count,
                    target_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM person_instances WHERE id = ?",
                (target_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError("merged person instance could not be loaded")
        return _person_instance_from_row(row)

    def get_instance(self, instance_id: str) -> PersonInstanceRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                "SELECT * FROM person_instances WHERE id = ?",
                (instance_id,),
            ).fetchone()
        return _person_instance_from_row(row) if row is not None else None

    def get_by_track(
        self,
        analysis_run_id: str,
        track_id: int,
    ) -> PersonInstanceRecord | None:
        if track_id < 1:
            raise ValueError("track_id must be at least 1")
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT instance.*
                FROM track_identity_links AS link
                JOIN person_instances AS instance ON instance.id = link.person_instance_id
                WHERE link.analysis_run_id = ? AND link.track_id = ?
                """,
                (analysis_run_id, track_id),
            ).fetchone()
        return _person_instance_from_row(row) if row is not None else None

    def get_by_identity(
        self,
        analysis_run_id: str,
        identity_id: str,
    ) -> PersonInstanceRecord | None:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT * FROM person_instances
                WHERE analysis_run_id = ? AND identity_id = ?
                """,
                (analysis_run_id, identity_id),
            ).fetchone()
        return _person_instance_from_row(row) if row is not None else None

    def list_stitch_candidates(
        self,
        *,
        analysis_run_id: str,
        candidate_identity_id: str,
        observed_at_seconds: float,
        max_gap_seconds: float,
    ) -> tuple[PersonInstanceRecord, ...]:
        earliest = max(0.0, observed_at_seconds - max_gap_seconds)
        with closing(connect_database(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT * FROM person_instances
                WHERE analysis_run_id = ?
                    AND (identity_id = ? OR best_candidate_identity_id = ?)
                    AND last_seen_timestamp_seconds BETWEEN ? AND ?
                ORDER BY last_seen_timestamp_seconds DESC, best_similarity DESC, id
                """,
                (
                    analysis_run_id,
                    candidate_identity_id,
                    candidate_identity_id,
                    earliest,
                    observed_at_seconds,
                ),
            ).fetchall()
        return tuple(_person_instance_from_row(row) for row in rows)

    def list_instances(
        self,
        *,
        analysis_run_id: str,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        status: str | None = None,
        identity_id: str | None = None,
    ) -> PersonInstancePage:
        _validate_pagination(page, limit)
        clauses = ["analysis_run_id = ?"]
        parameters: list[object] = [analysis_run_id]
        if status is not None:
            if status not in {"matched", "unknown"}:
                raise ValueError("status must be matched or unknown")
            clauses.append("status = ?")
            parameters.append(status)
        if identity_id is not None:
            clauses.append("identity_id = ?")
            parameters.append(identity_id)
        where_sql = " WHERE " + " AND ".join(clauses)
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM person_instances{where_sql}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM person_instances{where_sql}
                ORDER BY last_seen_timestamp_seconds DESC, id
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()
        return PersonInstancePage(
            items=tuple(_person_instance_from_row(row) for row in rows),
            total=total,
        )

    def list_links(
        self,
        person_instance_id: str,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> TrackIdentityLinkPage:
        _validate_pagination(page, limit)
        offset = (page - 1) * limit
        with closing(connect_database(self.database_path)) as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM track_identity_links WHERE person_instance_id = ?",
                    (person_instance_id,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT * FROM track_identity_links
                WHERE person_instance_id = ?
                ORDER BY first_seen_timestamp_seconds, track_id
                LIMIT ? OFFSET ?
                """,
                (person_instance_id, limit, offset),
            ).fetchall()
        return TrackIdentityLinkPage(
            items=tuple(_track_link_from_row(row) for row in rows),
            total=total,
        )

    def summarize_run(self, analysis_run_id: str) -> PersonInstanceRunSummary:
        with closing(connect_database(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_instances,
                    SUM(CASE WHEN status = 'matched' THEN 1 ELSE 0 END)
                        AS matched_instances,
                    SUM(CASE WHEN status = 'unknown' THEN 1 ELSE 0 END)
                        AS unknown_instances,
                    COALESCE(SUM(track_count), 0) AS linked_tracks
                FROM person_instances
                WHERE analysis_run_id = ?
                """,
                (analysis_run_id,),
            ).fetchone()
        return PersonInstanceRunSummary(
            analysis_run_id=analysis_run_id,
            total_instances=int(row["total_instances"]),
            matched_instances=int(row["matched_instances"] or 0),
            unknown_instances=int(row["unknown_instances"] or 0),
            linked_tracks=int(row["linked_tracks"] or 0),
        )


def _required_track_id(event: FaceMatchEventRecord) -> int:
    if event.track_id is None:
        raise ValueError("person-instance resolution requires a track_id")
    return event.track_id


def _validate_link(linked_by: str, confidence: float) -> None:
    if linked_by not in {
        "face_match",
        "identity_match",
        "candidate_stitch",
        "unknown_observation",
    }:
        raise ValueError("unsupported track identity link reason")
    if not -1 <= confidence <= 1:
        raise ValueError("link confidence must be between -1 and 1")


def _upsert_link(
    connection: sqlite3.Connection,
    instance_id: str,
    event: FaceMatchEventRecord,
    track_id: int,
    linked_by: str,
    confidence: float,
) -> None:
    existing = connection.execute(
        """
        SELECT person_instance_id FROM track_identity_links
        WHERE analysis_run_id = ? AND track_id = ?
        """,
        (event.analysis_run_id, track_id),
    ).fetchone()
    if existing is not None and str(existing["person_instance_id"]) != instance_id:
        raise ValueError("track_id is already linked to a different person instance")
    connection.execute(
        """
        INSERT INTO track_identity_links (
            analysis_run_id, track_id, person_instance_id, linked_by, confidence,
            first_sample_index, last_sample_index,
            first_seen_timestamp_seconds, last_seen_timestamp_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (analysis_run_id, track_id) DO UPDATE SET
            linked_by = CASE
                WHEN excluded.linked_by IN ('face_match', 'identity_match')
                    THEN excluded.linked_by
                ELSE track_identity_links.linked_by
            END,
            confidence = MAX(track_identity_links.confidence, excluded.confidence),
            first_sample_index = MIN(
                track_identity_links.first_sample_index, excluded.first_sample_index
            ),
            last_sample_index = MAX(
                track_identity_links.last_sample_index, excluded.last_sample_index
            ),
            first_seen_timestamp_seconds = MIN(
                track_identity_links.first_seen_timestamp_seconds,
                excluded.first_seen_timestamp_seconds
            ),
            last_seen_timestamp_seconds = MAX(
                track_identity_links.last_seen_timestamp_seconds,
                excluded.last_seen_timestamp_seconds
            ),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            event.analysis_run_id,
            track_id,
            instance_id,
            linked_by,
            confidence,
            event.sample_index,
            event.sample_index,
            event.source_timestamp_seconds,
            event.source_timestamp_seconds,
        ),
    )


def _assign_event(
    connection: sqlite3.Connection,
    event: FaceMatchEventRecord,
    instance_id: str,
) -> None:
    cursor = connection.execute(
        """
        UPDATE face_match_events SET person_instance_id = ?
        WHERE id = ? AND analysis_run_id = ?
        """,
        (instance_id, event.id, event.analysis_run_id),
    )
    if cursor.rowcount != 1:
        raise LookupError(f"face match event does not exist: {event.id}")


def _person_instance_from_row(row: sqlite3.Row) -> PersonInstanceRecord:
    return PersonInstanceRecord(
        id=str(row["id"]),
        analysis_run_id=str(row["analysis_run_id"]),
        status=str(row["status"]),
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
        first_sample_index=int(row["first_sample_index"]),
        last_sample_index=int(row["last_sample_index"]),
        first_seen_timestamp_seconds=float(row["first_seen_timestamp_seconds"]),
        last_seen_timestamp_seconds=float(row["last_seen_timestamp_seconds"]),
        last_face_x=int(row["last_face_x"]),
        last_face_y=int(row["last_face_y"]),
        last_face_width=int(row["last_face_width"]),
        last_face_height=int(row["last_face_height"]),
        track_count=int(row["track_count"]),
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _track_link_from_row(row: sqlite3.Row) -> TrackIdentityLinkRecord:
    return TrackIdentityLinkRecord(
        id=int(row["id"]),
        analysis_run_id=str(row["analysis_run_id"]),
        track_id=int(row["track_id"]),
        person_instance_id=str(row["person_instance_id"]),
        linked_by=str(row["linked_by"]),
        confidence=float(row["confidence"]),
        first_sample_index=int(row["first_sample_index"]),
        last_sample_index=int(row["last_sample_index"]),
        first_seen_timestamp_seconds=float(row["first_seen_timestamp_seconds"]),
        last_seen_timestamp_seconds=float(row["last_seen_timestamp_seconds"]),
        created_at=_datetime_from_text(str(row["created_at"])),
        updated_at=_datetime_from_text(str(row["updated_at"])),
    )


def _datetime_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _validate_pagination(page: int, limit: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


__all__ = [
    "PersonInstancePage",
    "PersonInstanceRecord",
    "PersonInstanceRepository",
    "PersonInstanceRunSummary",
    "TrackIdentityLinkPage",
    "TrackIdentityLinkRecord",
]
