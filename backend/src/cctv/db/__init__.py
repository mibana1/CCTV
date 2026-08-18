"""Database connections, schemas, repositories, and migrations."""

from cctv.db.database import (
    DatabaseHealth,
    DatabaseState,
    check_database_health,
    connect_database,
    initialize_database,
)
from cctv.db.detections import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AnalysisRunPage,
    AnalysisRunRecord,
    AnalysisRunStatus,
    DetectionPage,
    DetectionRecord,
    DetectionRepository,
    TrackObservationPage,
    TrackObservationRecord,
    TrackPage,
    TrackRecord,
)
from cctv.db.rules import (
    RuleEventPage,
    RuleEventRecord,
    RulePage,
    RuleRecord,
    RuleRepository,
)

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "AnalysisRunPage",
    "AnalysisRunRecord",
    "AnalysisRunStatus",
    "DatabaseHealth",
    "DatabaseState",
    "DetectionPage",
    "DetectionRecord",
    "DetectionRepository",
    "RuleEventPage",
    "RuleEventRecord",
    "RulePage",
    "RuleRecord",
    "RuleRepository",
    "TrackObservationPage",
    "TrackObservationRecord",
    "TrackPage",
    "TrackRecord",
    "check_database_health",
    "connect_database",
    "initialize_database",
]
