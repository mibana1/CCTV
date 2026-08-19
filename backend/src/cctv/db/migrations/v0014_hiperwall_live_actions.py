"""Upgrade simulated display actions into a durable LIVE-capable queue."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_display_actions_type_created")
    connection.execute("DROP INDEX IF EXISTS idx_display_actions_status_created")
    connection.execute("ALTER TABLE display_actions RENAME TO display_actions_v0008")
    connection.execute(
        """
        CREATE TABLE display_actions (
            id TEXT PRIMARY KEY,
            rule_event_id TEXT NOT NULL
                REFERENCES rule_events(id) ON DELETE CASCADE,
            action_type TEXT NOT NULL
                CHECK (action_type IN ('open_source', 'restore_layout')),
            mode TEXT NOT NULL CHECK (mode IN ('dry_run', 'live')),
            status TEXT NOT NULL CHECK (
                status IN ('simulated', 'pending', 'processing', 'retry', 'succeeded', 'failed')
            ),
            request_json TEXT NOT NULL CHECK (length(request_json) > 0),
            result_json TEXT NOT NULL CHECK (length(result_json) > 0),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            available_at TEXT NOT NULL,
            last_attempt_at TEXT,
            completed_at TEXT,
            last_error_code TEXT,
            last_error_message TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        INSERT INTO display_actions (
            id, rule_event_id, action_type, mode, status,
            request_json, result_json, available_at, created_at, updated_at
        )
        SELECT
            id, rule_event_id, action_type, mode, status,
            request_json, result_json, created_at, created_at, created_at
        FROM display_actions_v0008
        """
    )
    connection.execute("DROP TABLE display_actions_v0008")
    connection.execute(
        """
        CREATE INDEX idx_display_actions_type_created
        ON display_actions (action_type, created_at, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_display_actions_status_available
        ON display_actions (status, available_at, created_at, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_display_actions_rule_event
        ON display_actions (rule_event_id, created_at, id)
        """
    )
