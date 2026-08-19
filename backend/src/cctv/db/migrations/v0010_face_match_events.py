"""Persist privacy-safe face recognition decisions for analysis runs."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Create face-match event storage and run-oriented query indexes."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS face_match_events (
            id INTEGER PRIMARY KEY,
            analysis_run_id TEXT NOT NULL
                REFERENCES analysis_runs(id) ON DELETE CASCADE,
            source_index INTEGER NOT NULL CHECK (source_index >= 0),
            sample_index INTEGER NOT NULL CHECK (sample_index >= 0),
            source_timestamp_seconds REAL NOT NULL
                CHECK (source_timestamp_seconds >= 0),
            face_index INTEGER NOT NULL CHECK (face_index >= 0),
            track_id INTEGER CHECK (track_id IS NULL OR track_id > 0),
            face_x INTEGER NOT NULL CHECK (face_x >= 0),
            face_y INTEGER NOT NULL CHECK (face_y >= 0),
            face_width INTEGER NOT NULL CHECK (face_width > 0),
            face_height INTEGER NOT NULL CHECK (face_height > 0),
            detection_confidence REAL NOT NULL
                CHECK (detection_confidence >= 0 AND detection_confidence <= 1),
            match_status TEXT NOT NULL CHECK (match_status IN ('matched', 'unknown')),
            rejection_reason TEXT
                CHECK (rejection_reason IS NULL OR rejection_reason IN (
                    'below_threshold', 'ambiguous'
                )),
            identity_id TEXT,
            external_id TEXT,
            display_name TEXT,
            best_candidate_identity_id TEXT NOT NULL,
            best_candidate_external_id TEXT,
            best_similarity REAL NOT NULL
                CHECK (best_similarity >= -1 AND best_similarity <= 1),
            second_best_similarity REAL
                CHECK (
                    second_best_similarity IS NULL
                    OR (second_best_similarity >= -1 AND second_best_similarity <= 1)
                ),
            similarity_threshold REAL NOT NULL
                CHECK (similarity_threshold > 0 AND similarity_threshold <= 1),
            minimum_margin REAL NOT NULL
                CHECK (minimum_margin >= 0 AND minimum_margin <= 1),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (
                (match_status = 'matched' AND rejection_reason IS NULL AND identity_id IS NOT NULL)
                OR (match_status = 'unknown' AND rejection_reason IS NOT NULL AND identity_id IS NULL)
            )
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_face_match_events_run_sample
        ON face_match_events (analysis_run_id, sample_index, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_face_match_events_run_status
        ON face_match_events (analysis_run_id, match_status, best_similarity DESC, id)
        """
    )

