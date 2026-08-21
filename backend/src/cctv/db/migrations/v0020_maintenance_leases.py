"""Add cross-backend maintenance leases and retention query indexes."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    connection.execute(
        """
        CREATE TABLE maintenance_leases (
            name TEXT PRIMARY KEY CHECK (length(name) BETWEEN 1 AND 64),
            owner_id TEXT NOT NULL CHECK (length(owner_id) BETWEEN 1 AND 128),
            expires_at TEXT NOT NULL CHECK (length(expires_at) > 0),
            updated_at TEXT NOT NULL CHECK (length(updated_at) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_maintenance_leases_expiry
        ON maintenance_leases (expires_at, name)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_analyzed_frames_created_id
        ON analyzed_frames (created_at, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_rule_events_created_frame
        ON rule_events (created_at, analyzed_frame_id, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_tracks_updated_run_track
        ON tracks (updated_at, analysis_run_id, track_id)
        """
    )
