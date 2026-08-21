"""Keep rule history while allowing dashboard rules to be removed."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Add a tombstone timestamp used to hide deleted rules from active queries."""
    connection.execute("ALTER TABLE rules ADD COLUMN deleted_at TEXT")
    connection.execute(
        """
        CREATE INDEX idx_rules_visible_source_enabled
        ON rules (deleted_at, source_name, enabled, rule_type, id)
        """
    )


__all__ = ["apply"]
