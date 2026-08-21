"""Add expiring camera ownership leases for continuous analysis workers."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    connection.execute(
        """
        CREATE TABLE analysis_worker_leases (
            camera_id INTEGER PRIMARY KEY
                REFERENCES cameras(id) ON DELETE CASCADE,
            owner_id TEXT NOT NULL CHECK (length(owner_id) BETWEEN 1 AND 128),
            expires_at TEXT NOT NULL CHECK (length(expires_at) > 0),
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_analysis_worker_leases_expiry
        ON analysis_worker_leases (expires_at, camera_id)
        """
    )
