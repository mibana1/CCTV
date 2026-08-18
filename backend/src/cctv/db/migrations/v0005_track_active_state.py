"""Add current tracker-active state to object-track lifetimes."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Add an inactive-safe state column and indexes for filtered track queries."""
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(tracks)").fetchall()}
    if "is_active" not in columns:
        connection.execute(
            """
            ALTER TABLE tracks
            ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0
                CHECK (is_active IN (0, 1))
            """
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tracks_active_class_time
        ON tracks (
            is_active,
            class_name,
            first_seen_timestamp_seconds,
            last_seen_timestamp_seconds,
            analysis_run_id,
            track_id
        )
        """
    )
