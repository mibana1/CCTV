"""Add per-analysis-run object tracking identity to detections."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Add a nullable track ID so historical untracked rows remain valid."""
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(detections)").fetchall()
    }
    if "track_id" not in columns:
        connection.execute(
            "ALTER TABLE detections ADD COLUMN track_id INTEGER CHECK (track_id > 0)"
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_track
        ON detections (track_id, analyzed_frame_id, id)
        """
    )
