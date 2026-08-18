"""Create extensible rule definitions and track-linked event history."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Create generic JSON-configured rules without fixing evaluator types in SQL."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL CHECK (length(name) > 0),
            source_name TEXT NOT NULL CHECK (length(source_name) > 0),
            rule_type TEXT NOT NULL CHECK (length(rule_type) > 0),
            class_name TEXT CHECK (class_name IS NULL OR length(class_name) > 0),
            geometry_json TEXT NOT NULL CHECK (length(geometry_json) > 0),
            parameters_json TEXT NOT NULL DEFAULT '{}'
                CHECK (length(parameters_json) > 0),
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source_name, name)
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_analyzed_frames_run_id
        ON analyzed_frames (analysis_run_id, id)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_events (
            id TEXT PRIMARY KEY,
            rule_id TEXT NOT NULL REFERENCES rules(id) ON DELETE RESTRICT,
            analysis_run_id TEXT NOT NULL
                REFERENCES analysis_runs(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL CHECK (track_id > 0),
            analyzed_frame_id INTEGER NOT NULL
                REFERENCES analyzed_frames(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL CHECK (length(event_type) > 0),
            event_state TEXT NOT NULL
                CHECK (event_state IN ('started', 'ended', 'occurred')),
            occurred_at_seconds REAL NOT NULL CHECK (occurred_at_seconds >= 0),
            class_name TEXT NOT NULL CHECK (length(class_name) > 0),
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            payload_json TEXT NOT NULL DEFAULT '{}' CHECK (length(payload_json) > 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_run_id, analyzed_frame_id)
                REFERENCES analyzed_frames(analysis_run_id, id) ON DELETE CASCADE,
            FOREIGN KEY (analysis_run_id, track_id)
                REFERENCES tracks(analysis_run_id, track_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rules_source_enabled
        ON rules (source_name, enabled, rule_type, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rule_events_rule_time
        ON rule_events (rule_id, occurred_at_seconds, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rule_events_run_track_time
        ON rule_events (analysis_run_id, track_id, occurred_at_seconds, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rule_events_type_time
        ON rule_events (event_type, occurred_at_seconds, id)
        """
    )
