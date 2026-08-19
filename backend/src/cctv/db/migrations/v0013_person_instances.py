"""Group fragmented object tracks into session-scoped person instances."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Create person-instance links and associate face events without storing embeddings."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS person_instances (
            id TEXT PRIMARY KEY,
            analysis_run_id TEXT NOT NULL
                REFERENCES analysis_runs(id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('matched', 'unknown')),
            identity_id TEXT,
            external_id TEXT,
            display_name TEXT,
            best_candidate_identity_id TEXT NOT NULL,
            best_candidate_external_id TEXT,
            best_similarity REAL NOT NULL CHECK (best_similarity BETWEEN -1 AND 1),
            first_sample_index INTEGER NOT NULL CHECK (first_sample_index >= 0),
            last_sample_index INTEGER NOT NULL CHECK (last_sample_index >= first_sample_index),
            first_seen_timestamp_seconds REAL NOT NULL
                CHECK (first_seen_timestamp_seconds >= 0),
            last_seen_timestamp_seconds REAL NOT NULL
                CHECK (last_seen_timestamp_seconds >= first_seen_timestamp_seconds),
            last_face_x INTEGER NOT NULL CHECK (last_face_x >= 0),
            last_face_y INTEGER NOT NULL CHECK (last_face_y >= 0),
            last_face_width INTEGER NOT NULL CHECK (last_face_width > 0),
            last_face_height INTEGER NOT NULL CHECK (last_face_height > 0),
            track_count INTEGER NOT NULL DEFAULT 0 CHECK (track_count >= 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (
                (status = 'matched' AND identity_id IS NOT NULL)
                OR (status = 'unknown' AND identity_id IS NULL)
            )
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_person_instances_run_identity
        ON person_instances (analysis_run_id, identity_id)
        WHERE identity_id IS NOT NULL
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_person_instances_run_status
        ON person_instances (
            analysis_run_id, status, last_seen_timestamp_seconds DESC, id
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_person_instances_stitch_candidate
        ON person_instances (
            analysis_run_id, best_candidate_identity_id,
            last_seen_timestamp_seconds DESC, id
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS track_identity_links (
            id INTEGER PRIMARY KEY,
            analysis_run_id TEXT NOT NULL
                REFERENCES analysis_runs(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL CHECK (track_id > 0),
            person_instance_id TEXT NOT NULL
                REFERENCES person_instances(id) ON DELETE CASCADE,
            linked_by TEXT NOT NULL CHECK (linked_by IN (
                'face_match', 'identity_match', 'candidate_stitch', 'unknown_observation'
            )),
            confidence REAL NOT NULL CHECK (confidence BETWEEN -1 AND 1),
            first_sample_index INTEGER NOT NULL CHECK (first_sample_index >= 0),
            last_sample_index INTEGER NOT NULL CHECK (last_sample_index >= first_sample_index),
            first_seen_timestamp_seconds REAL NOT NULL
                CHECK (first_seen_timestamp_seconds >= 0),
            last_seen_timestamp_seconds REAL NOT NULL
                CHECK (last_seen_timestamp_seconds >= first_seen_timestamp_seconds),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (analysis_run_id, track_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_track_identity_links_instance
        ON track_identity_links (
            person_instance_id, first_seen_timestamp_seconds, track_id
        )
        """
    )

    face_match_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(face_match_events)")
    }
    if "person_instance_id" not in face_match_columns:
        connection.execute(
            """
            ALTER TABLE face_match_events
            ADD COLUMN person_instance_id TEXT
                REFERENCES person_instances(id) ON DELETE SET NULL
            """
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_face_match_events_person_instance
        ON face_match_events (person_instance_id, sample_index, id)
        """
    )

