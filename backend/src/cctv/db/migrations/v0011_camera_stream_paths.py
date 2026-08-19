"""Add MediaMTX stream paths to the camera registry."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Add and safely backfill the browser-preview stream identifier."""
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(cameras)").fetchall()
    }
    if "stream_path" not in columns:
        connection.execute("ALTER TABLE cameras ADD COLUMN stream_path TEXT")

    connection.execute(
        """
        UPDATE cameras
        SET stream_path = CASE
            WHEN id = (SELECT MIN(id) FROM cameras) THEN 'camera'
            ELSE 'camera-' || id
        END
        WHERE stream_path IS NULL OR TRIM(stream_path) = ''
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cameras_stream_path
        ON cameras (stream_path COLLATE NOCASE)
        WHERE stream_path IS NOT NULL
        """
    )

