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
    "check_database_health",
    "connect_database",
    "initialize_database",
]
