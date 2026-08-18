"""Create auditable Hiperwall DRY RUN display actions."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Store one simulated Hiperwall operation for each handled rule event."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS display_actions (
            id TEXT PRIMARY KEY,
            rule_event_id TEXT NOT NULL UNIQUE
                REFERENCES rule_events(id) ON DELETE CASCADE,
            action_type TEXT NOT NULL
                CHECK (action_type IN ('open_source', 'restore_layout')),
            mode TEXT NOT NULL CHECK (mode = 'dry_run'),
            status TEXT NOT NULL CHECK (status = 'simulated'),
            request_json TEXT NOT NULL CHECK (length(request_json) > 0),
            result_json TEXT NOT NULL CHECK (length(result_json) > 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_display_actions_type_created
        ON display_actions (action_type, created_at, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_display_actions_status_created
        ON display_actions (status, created_at, id)
        """
    )
