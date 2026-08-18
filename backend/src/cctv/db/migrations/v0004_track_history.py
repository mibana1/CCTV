"""Create normalized object-track lifetimes and observation history."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Create track history tables and backfill previously tracked detections."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY,
            analysis_run_id TEXT NOT NULL
                REFERENCES analysis_runs(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL CHECK (track_id > 0),
            class_id INTEGER NOT NULL CHECK (class_id >= 0),
            class_name TEXT NOT NULL CHECK (length(class_name) > 0),
            first_sample_index INTEGER NOT NULL CHECK (first_sample_index >= 0),
            last_sample_index INTEGER NOT NULL CHECK (last_sample_index >= first_sample_index),
            first_seen_timestamp_seconds REAL NOT NULL
                CHECK (first_seen_timestamp_seconds >= 0),
            last_seen_timestamp_seconds REAL NOT NULL
                CHECK (last_seen_timestamp_seconds >= first_seen_timestamp_seconds),
            observation_count INTEGER NOT NULL CHECK (observation_count > 0),
            max_confidence REAL NOT NULL CHECK (max_confidence >= 0 AND max_confidence <= 1),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (analysis_run_id, track_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS track_observations (
            id INTEGER PRIMARY KEY,
            analysis_run_id TEXT NOT NULL,
            track_id INTEGER NOT NULL CHECK (track_id > 0),
            analyzed_frame_id INTEGER NOT NULL
                REFERENCES analyzed_frames(id) ON DELETE CASCADE,
            detection_id INTEGER NOT NULL UNIQUE
                REFERENCES detections(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_run_id, track_id)
                REFERENCES tracks(analysis_run_id, track_id) ON DELETE CASCADE,
            UNIQUE (analysis_run_id, track_id, analyzed_frame_id)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO tracks (
            analysis_run_id, track_id, class_id, class_name,
            first_sample_index, last_sample_index,
            first_seen_timestamp_seconds, last_seen_timestamp_seconds,
            observation_count, max_confidence
        )
        SELECT
            frame.analysis_run_id,
            detection.track_id,
            MIN(detection.class_id),
            MIN(detection.class_name),
            MIN(frame.sample_index),
            MAX(frame.sample_index),
            MIN(frame.source_timestamp_seconds),
            MAX(frame.source_timestamp_seconds),
            COUNT(*),
            MAX(detection.confidence)
        FROM detections AS detection
        JOIN analyzed_frames AS frame ON frame.id = detection.analyzed_frame_id
        WHERE detection.track_id IS NOT NULL
        GROUP BY frame.analysis_run_id, detection.track_id
        ON CONFLICT (analysis_run_id, track_id) DO NOTHING
        """
    )
    connection.execute(
        """
        INSERT INTO track_observations (
            analysis_run_id, track_id, analyzed_frame_id, detection_id
        )
        SELECT
            frame.analysis_run_id,
            detection.track_id,
            frame.id,
            detection.id
        FROM detections AS detection
        JOIN analyzed_frames AS frame ON frame.id = detection.analyzed_frame_id
        WHERE detection.track_id IS NOT NULL
        ON CONFLICT DO NOTHING
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tracks_run_class
        ON tracks (analysis_run_id, class_name, track_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tracks_run_last_seen
        ON tracks (analysis_run_id, last_seen_timestamp_seconds DESC, track_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_track_observations_track
        ON track_observations (analysis_run_id, track_id, analyzed_frame_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_track_observations_frame
        ON track_observations (analyzed_frame_id, id)
        """
    )
