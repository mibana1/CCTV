"""Ordered SQLite schema migrations."""

from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Connection

from cctv.db.migrations.v0001_initial import apply as apply_v0001
from cctv.db.migrations.v0002_detection_results import apply as apply_v0002


@dataclass(frozen=True)
class Migration:
    """One forward-only, repeatable schema migration."""

    version: int
    name: str
    apply: Callable[[Connection], None]


MIGRATIONS = (
    Migration(version=1, name="initial_camera_schema", apply=apply_v0001),
    Migration(version=2, name="detection_result_schema", apply=apply_v0002),
)

LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version

__all__ = ["LATEST_SCHEMA_VERSION", "MIGRATIONS", "Migration"]
