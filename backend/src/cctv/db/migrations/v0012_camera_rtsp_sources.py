"""Add encrypted RTSP source metadata and provisioning state to cameras."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Extend camera registrations without exposing source credentials."""
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(cameras)").fetchall()
    }
    additions = {
        "rtsp_endpoint": "TEXT",
        "rtsp_source_ciphertext": "TEXT",
        "source_on_demand": "INTEGER NOT NULL DEFAULT 1",
        "provisioning_status": "TEXT NOT NULL DEFAULT 'external'",
        "last_sync_error": "TEXT",
        "last_synced_at": "TEXT",
    }
    for name, definition in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE cameras ADD COLUMN {name} {definition}")

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cameras_provisioning_status
        ON cameras (provisioning_status, enabled)
        """
    )

