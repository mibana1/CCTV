"""Create registered identities and model-versioned embedding storage."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Create model-neutral identity records with normalized float32 vectors."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS identities (
            id TEXT PRIMARY KEY,
            external_id TEXT COLLATE NOCASE UNIQUE,
            display_name TEXT NOT NULL CHECK (length(display_name) > 0),
            description TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
                CHECK (length(metadata_json) > 0),
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (external_id IS NULL OR length(external_id) > 0),
            CHECK (description IS NULL OR length(description) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS identity_embeddings (
            id TEXT PRIMARY KEY,
            identity_id TEXT NOT NULL
                REFERENCES identities(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL COLLATE NOCASE CHECK (length(model_name) > 0),
            model_version TEXT NOT NULL COLLATE NOCASE CHECK (length(model_version) > 0),
            dimension INTEGER NOT NULL CHECK (dimension BETWEEN 2 AND 4096),
            dtype TEXT NOT NULL DEFAULT 'float32' CHECK (dtype = 'float32'),
            normalized INTEGER NOT NULL DEFAULT 1 CHECK (normalized = 1),
            embedding_sha256 TEXT NOT NULL CHECK (length(embedding_sha256) = 64),
            vector_blob BLOB NOT NULL CHECK (length(vector_blob) = dimension * 4),
            source_reference TEXT,
            quality_score REAL CHECK (
                quality_score IS NULL OR quality_score BETWEEN 0 AND 1
            ),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (source_reference IS NULL OR length(source_reference) > 0),
            UNIQUE (
                identity_id, model_name, model_version, embedding_sha256
            )
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_identities_enabled_name
        ON identities (enabled, display_name COLLATE NOCASE, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_identity_embeddings_identity_model
        ON identity_embeddings (
            identity_id, model_name COLLATE NOCASE, model_version COLLATE NOCASE, id
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_identity_embeddings_matching
        ON identity_embeddings (
            model_name COLLATE NOCASE, model_version COLLATE NOCASE,
            dimension, identity_id
        )
        """
    )
