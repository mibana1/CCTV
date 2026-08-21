"""Ordered SQLite schema migrations."""

from collections.abc import Callable
from dataclasses import dataclass
from sqlite3 import Connection

from cctv.db.migrations.v0001_initial import apply as apply_v0001
from cctv.db.migrations.v0002_detection_results import apply as apply_v0002
from cctv.db.migrations.v0003_detection_tracking import apply as apply_v0003
from cctv.db.migrations.v0004_track_history import apply as apply_v0004
from cctv.db.migrations.v0005_track_active_state import apply as apply_v0005
from cctv.db.migrations.v0006_rule_engine import apply as apply_v0006
from cctv.db.migrations.v0007_detector_type import apply as apply_v0007
from cctv.db.migrations.v0008_display_actions import apply as apply_v0008
from cctv.db.migrations.v0009_identity_registry import apply as apply_v0009
from cctv.db.migrations.v0010_face_match_events import apply as apply_v0010
from cctv.db.migrations.v0011_camera_stream_paths import apply as apply_v0011
from cctv.db.migrations.v0012_camera_rtsp_sources import apply as apply_v0012
from cctv.db.migrations.v0013_person_instances import apply as apply_v0013
from cctv.db.migrations.v0014_hiperwall_live_actions import apply as apply_v0014
from cctv.db.migrations.v0015_camera_always_connected import apply as apply_v0015
from cctv.db.migrations.v0016_candidate_free_face_events import apply as apply_v0016
from cctv.db.migrations.v0017_rule_soft_delete import apply as apply_v0017
from cctv.db.migrations.v0018_analysis_worker_leases import apply as apply_v0018
from cctv.db.migrations.v0019_hiperwall_display_states import apply as apply_v0019
from cctv.db.migrations.v0020_maintenance_leases import apply as apply_v0020


@dataclass(frozen=True)
class Migration:
    """One forward-only, repeatable schema migration."""

    version: int
    name: str
    apply: Callable[[Connection], None]


MIGRATIONS = (
    Migration(version=1, name="initial_camera_schema", apply=apply_v0001),
    Migration(version=2, name="detection_result_schema", apply=apply_v0002),
    Migration(version=3, name="detection_tracking_schema", apply=apply_v0003),
    Migration(version=4, name="track_history_schema", apply=apply_v0004),
    Migration(version=5, name="track_active_state_schema", apply=apply_v0005),
    Migration(version=6, name="rule_engine_schema", apply=apply_v0006),
    Migration(version=7, name="detector_type", apply=apply_v0007),
    Migration(version=8, name="display_actions", apply=apply_v0008),
    Migration(version=9, name="identity_registry", apply=apply_v0009),
    Migration(version=10, name="face_match_events", apply=apply_v0010),
    Migration(version=11, name="camera_stream_paths", apply=apply_v0011),
    Migration(version=12, name="camera_rtsp_sources", apply=apply_v0012),
    Migration(version=13, name="person_instances", apply=apply_v0013),
    Migration(version=14, name="hiperwall_live_actions", apply=apply_v0014),
    Migration(version=15, name="camera_always_connected", apply=apply_v0015),
    Migration(version=16, name="candidate_free_face_events", apply=apply_v0016),
    Migration(version=17, name="rule_soft_delete", apply=apply_v0017),
    Migration(version=18, name="analysis_worker_leases", apply=apply_v0018),
    Migration(version=19, name="hiperwall_display_states", apply=apply_v0019),
    Migration(version=20, name="maintenance_leases", apply=apply_v0020),
)

LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version

__all__ = ["LATEST_SCHEMA_VERSION", "MIGRATIONS", "Migration"]
