"""Record the interchangeable detector adapter used by each analysis run."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Backfill historical YOLO runs and require a detector registry key."""
    connection.execute(
        """
        ALTER TABLE analysis_runs
        ADD COLUMN detector_type TEXT NOT NULL DEFAULT 'yolo_onnx'
            CHECK (length(detector_type) > 0)
        """
    )
