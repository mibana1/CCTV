"""Lease-protected batched retention for database records and snapshot files."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import ceil
from pathlib import Path
from shutil import disk_usage
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import Settings, get_settings
from cctv.db import (
    DatabaseActivity,
    MaintenanceLeaseRepository,
    RetentionCounts,
    RetentionDatabaseMetrics,
    RetentionRepository,
    WalCheckpointResult,
    initialize_database,
)

logger = logging.getLogger(__name__)
_LEASE_NAME = "data_retention"

Clock = Callable[[], float]
NowFactory = Callable[[], datetime]
DiskFreeFactory = Callable[[Path], int]


class RetentionRunStatus(StrEnum):
    COMPLETED = "completed"
    SKIPPED = "skipped"


class VacuumRunStatus(StrEnum):
    DISABLED = "disabled"
    OUTSIDE_MAINTENANCE_WINDOW = "outside_maintenance_window"
    CHECKPOINT_BUSY = "checkpoint_busy"
    BELOW_THRESHOLD = "below_threshold"
    ACTIVE_WORK = "active_work"
    INSUFFICIENT_DISK_SPACE = "insufficient_disk_space"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class VacuumRunResult:
    status: VacuumRunStatus
    freelist_ratio: float
    freelist_count: int
    page_count: int
    active_work: DatabaseActivity | None = None
    required_free_bytes: int | None = None
    database_free_bytes: int | None = None
    backup_free_bytes: int | None = None
    backup_path: Path | None = None
    backup_bytes: int | None = None
    post_vacuum_checkpoint: WalCheckpointResult | None = None
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RetentionRunResult:
    status: RetentionRunStatus
    dry_run: bool
    lease_acquired: bool
    owner_id: str
    frame_cutoff_at: datetime
    audit_cutoff_at: datetime
    snapshot_cutoff_at: datetime
    database_batches: int
    snapshot_batches: int
    counts: RetentionCounts
    database_before: RetentionDatabaseMetrics | None
    database_after: RetentionDatabaseMetrics | None
    checkpoint: WalCheckpointResult | None
    vacuum: VacuumRunResult | None
    elapsed_seconds: float


class RetentionWorker:
    """Periodically remove expired data while one Backend owns the maintenance lease."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository: RetentionRepository | None = None,
        lease_repository: MaintenanceLeaseRepository | None = None,
        owner_id: str | None = None,
        monotonic_clock: Clock = monotonic,
        now_factory: NowFactory | None = None,
        disk_free_factory: DiskFreeFactory | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or RetentionRepository(settings.database_path)
        self.lease_repository = lease_repository or MaintenanceLeaseRepository(
            settings.database_path
        )
        self.owner_id = owner_id or str(uuid4())
        self._monotonic = monotonic_clock
        self._now = now_factory or (lambda: datetime.now(UTC))
        self._disk_free = disk_free_factory or _disk_free_bytes
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="cctv-retention", daemon=True)
        self._thread.start()
        logger.info(
            "Retention worker started",
            extra={
                "event": "retention_worker_started",
                "dry_run": self.settings.retention_dry_run,
                "interval_seconds": self.settings.retention_interval_seconds,
            },
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info(
            "Retention worker stopped",
            extra={"event": "retention_worker_stopped"},
        )

    def run_once(self, *, dry_run: bool | None = None) -> RetentionRunResult:
        started_at = self._monotonic()
        run_at = _utc(self._now())
        effective_dry_run = self.settings.retention_dry_run if dry_run is None else dry_run
        frame_cutoff = run_at - timedelta(days=self.settings.retention_frame_days)
        audit_cutoff = run_at - timedelta(days=self.settings.retention_audit_days)
        snapshot_cutoff = run_at - timedelta(days=self.settings.retention_snapshot_days)
        before: RetentionDatabaseMetrics | None = None

        logger.info(
            "Retention cleanup started",
            extra={
                "event": "retention_cleanup_started",
                "owner_id": self.owner_id,
                "dry_run": effective_dry_run,
                "frame_cutoff_at": frame_cutoff,
                "audit_cutoff_at": audit_cutoff,
                "snapshot_cutoff_at": snapshot_cutoff,
                "batch_size": self.settings.retention_batch_size,
                "max_batches": self.settings.retention_max_batches_per_run,
            },
        )
        try:
            acquired = self.lease_repository.acquire(
                _LEASE_NAME,
                self.owner_id,
                lease_seconds=self.settings.retention_lease_seconds,
                now=run_at,
            )
        except Exception as error:
            logger.exception(
                "Retention cleanup lease acquisition failed",
                extra={
                    "event": "retention_cleanup_failed",
                    "owner_id": self.owner_id,
                    "dry_run": effective_dry_run,
                    "error_type": type(error).__name__,
                    "elapsed_seconds": round(self._monotonic() - started_at, 6),
                },
            )
            raise
        if not acquired:
            result = RetentionRunResult(
                status=RetentionRunStatus.SKIPPED,
                dry_run=effective_dry_run,
                lease_acquired=False,
                owner_id=self.owner_id,
                frame_cutoff_at=frame_cutoff,
                audit_cutoff_at=audit_cutoff,
                snapshot_cutoff_at=snapshot_cutoff,
                database_batches=0,
                snapshot_batches=0,
                counts=RetentionCounts(),
                database_before=None,
                database_after=None,
                checkpoint=None,
                vacuum=None,
                elapsed_seconds=round(self._monotonic() - started_at, 6),
            )
            logger.info(
                "Retention cleanup skipped because another Backend owns the lease",
                extra={"event": "retention_cleanup_lease_unavailable", **asdict(result)},
            )
            return result

        try:
            before = self.repository.metrics()
            checkpoint: WalCheckpointResult | None = None
            vacuum: VacuumRunResult | None = None
            if effective_dry_run:
                maximum_items = (
                    self.settings.retention_batch_size
                    * self.settings.retention_max_batches_per_run
                )
                counts = self.repository.preview(
                    frame_cutoff=frame_cutoff,
                    audit_cutoff=audit_cutoff,
                    limit=maximum_items,
                )
                snapshots = _snapshot_candidates(
                    self.settings.snapshot_dir,
                    protected_root=self.settings.identity_photo_dir,
                    cutoff=snapshot_cutoff,
                    limit=maximum_items,
                )
                counts += _snapshot_counts(snapshots)
                database_batches = 0
                snapshot_batches = 0
            else:
                counts, database_batches = self._delete_database_batches(
                    frame_cutoff=frame_cutoff,
                    audit_cutoff=audit_cutoff,
                )
                snapshot_counts, snapshot_batches = self._delete_snapshot_batches(
                    cutoff=snapshot_cutoff
                )
                counts += snapshot_counts
                checkpoint, vacuum = self._maintain_database(run_at=run_at)

            after = self.repository.metrics()
            result = RetentionRunResult(
                status=RetentionRunStatus.COMPLETED,
                dry_run=effective_dry_run,
                lease_acquired=True,
                owner_id=self.owner_id,
                frame_cutoff_at=frame_cutoff,
                audit_cutoff_at=audit_cutoff,
                snapshot_cutoff_at=snapshot_cutoff,
                database_batches=database_batches,
                snapshot_batches=snapshot_batches,
                counts=counts,
                database_before=before,
                database_after=after,
                checkpoint=checkpoint,
                vacuum=vacuum,
                elapsed_seconds=round(self._monotonic() - started_at, 6),
            )
            logger.info(
                "Retention cleanup completed",
                extra={
                    "event": "retention_cleanup_completed",
                    **asdict(result),
                    "database_before_total_bytes": before.total_bytes,
                    "database_after_total_bytes": after.total_bytes,
                    "freelist_count_before": before.freelist_count,
                    "freelist_count_after": after.freelist_count,
                },
            )
            return result
        except Exception as error:
            after = _safe_metrics(self.repository)
            logger.exception(
                "Retention cleanup failed",
                extra={
                    "event": "retention_cleanup_failed",
                    "owner_id": self.owner_id,
                    "dry_run": effective_dry_run,
                    "error_type": type(error).__name__,
                    "elapsed_seconds": round(self._monotonic() - started_at, 6),
                    "database_before": asdict(before) if before is not None else None,
                    "database_after": asdict(after) if after is not None else None,
                },
            )
            raise
        finally:
            try:
                self.lease_repository.release(_LEASE_NAME, self.owner_id)
            except Exception:
                logger.exception(
                    "Retention maintenance lease could not be released",
                    extra={
                        "event": "retention_lease_release_failed",
                        "owner_id": self.owner_id,
                    },
                )

    def _delete_database_batches(
        self,
        *,
        frame_cutoff: datetime,
        audit_cutoff: datetime,
    ) -> tuple[RetentionCounts, int]:
        total = RetentionCounts()
        batches = 0
        for _ in range(self.settings.retention_max_batches_per_run):
            self._renew_lease()
            batch = self.repository.delete_batch(
                frame_cutoff=frame_cutoff,
                audit_cutoff=audit_cutoff,
                batch_size=self.settings.retention_batch_size,
            )
            if batch.analyzed_frames == 0 and batch.tracks == 0:
                break
            total += batch
            batches += 1
        return total, batches

    def _delete_snapshot_batches(self, *, cutoff: datetime) -> tuple[RetentionCounts, int]:
        total = RetentionCounts()
        batches = 0
        for _ in range(self.settings.retention_max_batches_per_run):
            self._renew_lease()
            candidates = _snapshot_candidates(
                self.settings.snapshot_dir,
                protected_root=self.settings.identity_photo_dir,
                cutoff=cutoff,
                limit=self.settings.retention_batch_size,
            )
            if not candidates:
                break
            batch = RetentionCounts()
            for path, size in candidates:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                batch += RetentionCounts(snapshots=1, snapshot_bytes=size)
            total += batch
            batches += 1
        return total, batches

    def _maintain_database(
        self,
        *,
        run_at: datetime,
    ) -> tuple[WalCheckpointResult | None, VacuumRunResult]:
        activity = self.repository.activity(now=run_at)
        in_maintenance_window = _in_maintenance_window(
            run_at,
            start_hour_utc=self.settings.retention_maintenance_window_start_hour_utc,
            duration_minutes=self.settings.retention_maintenance_window_duration_minutes,
        )
        checkpoint: WalCheckpointResult | None = None
        if self.settings.retention_checkpoint_enabled:
            checkpoint_mode = (
                "truncate"
                if self.settings.retention_truncate_checkpoint_enabled
                and in_maintenance_window
                and not activity.active
                else "passive"
            )
            self._renew_lease()
            checkpoint = self.repository.checkpoint(mode=checkpoint_mode)
            logger.info(
                "Retention WAL checkpoint completed",
                extra={
                    "event": "retention_wal_checkpoint_completed",
                    **asdict(checkpoint),
                    "in_maintenance_window": in_maintenance_window,
                    "active_work": asdict(activity),
                },
            )

        metrics = self.repository.metrics()
        vacuum = self._run_conditional_vacuum(
            run_at=run_at,
            metrics=metrics,
            activity=activity,
            in_maintenance_window=in_maintenance_window,
            checkpoint=checkpoint,
        )
        log_level = (
            logging.DEBUG if vacuum.status is VacuumRunStatus.DISABLED else logging.INFO
        )
        logger.log(
            log_level,
            "Retention conditional VACUUM evaluated",
            extra={"event": "retention_vacuum_evaluated", **asdict(vacuum)},
        )
        return checkpoint, vacuum

    def _run_conditional_vacuum(
        self,
        *,
        run_at: datetime,
        metrics: RetentionDatabaseMetrics,
        activity: DatabaseActivity,
        in_maintenance_window: bool,
        checkpoint: WalCheckpointResult | None,
    ) -> VacuumRunResult:
        started_at = self._monotonic()
        ratio = (
            metrics.freelist_count / metrics.page_count if metrics.page_count > 0 else 0.0
        )

        def result(status: VacuumRunStatus, **values: object) -> VacuumRunResult:
            return VacuumRunResult(
                status=status,
                freelist_ratio=round(ratio, 6),
                freelist_count=metrics.freelist_count,
                page_count=metrics.page_count,
                elapsed_seconds=round(self._monotonic() - started_at, 6),
                **values,  # type: ignore[arg-type]
            )

        if not self.settings.retention_vacuum_enabled:
            return result(VacuumRunStatus.DISABLED)
        if not in_maintenance_window:
            return result(VacuumRunStatus.OUTSIDE_MAINTENANCE_WINDOW)
        if checkpoint is not None and checkpoint.busy > 0:
            return result(VacuumRunStatus.CHECKPOINT_BUSY)
        if (
            metrics.freelist_count < self.settings.retention_vacuum_min_freelist_pages
            or ratio < self.settings.retention_vacuum_freelist_ratio_threshold
        ):
            return result(VacuumRunStatus.BELOW_THRESHOLD)
        if activity.active:
            return result(VacuumRunStatus.ACTIVE_WORK, active_work=activity)

        backup_dir = self.settings.retention_vacuum_backup_dir.expanduser().resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        required_free_bytes = ceil(
            metrics.database_bytes * self.settings.retention_vacuum_free_space_multiplier
        )
        database_free_bytes = self._disk_free(self.repository.database_path.parent)
        backup_free_bytes = self._disk_free(backup_dir)
        space = {
            "required_free_bytes": required_free_bytes,
            "database_free_bytes": database_free_bytes,
            "backup_free_bytes": backup_free_bytes,
        }
        if min(database_free_bytes, backup_free_bytes) < required_free_bytes:
            return result(VacuumRunStatus.INSUFFICIENT_DISK_SPACE, **space)

        backup_path = backup_dir / _backup_filename(run_at, owner_id=self.owner_id)
        self._renew_lease()
        completed_backup = self.repository.online_backup(backup_path)
        backup_bytes = completed_backup.stat().st_size
        logger.info(
            "Pre-VACUUM online backup completed",
            extra={
                "event": "retention_vacuum_backup_completed",
                "backup_path": completed_backup,
                "backup_bytes": backup_bytes,
                **space,
            },
        )

        latest_activity = self.repository.activity(now=_utc(self._now()))
        if latest_activity.active:
            return result(
                VacuumRunStatus.ACTIVE_WORK,
                active_work=latest_activity,
                backup_path=completed_backup,
                backup_bytes=backup_bytes,
                **space,
            )

        self._renew_lease()
        self.repository.vacuum()
        self._renew_lease()
        post_checkpoint = self.repository.checkpoint(mode="passive")
        return result(
            VacuumRunStatus.COMPLETED,
            active_work=latest_activity,
            backup_path=completed_backup,
            backup_bytes=backup_bytes,
            post_vacuum_checkpoint=post_checkpoint,
            **space,
        )

    def _renew_lease(self) -> None:
        if not self.lease_repository.acquire(
            _LEASE_NAME,
            self.owner_id,
            lease_seconds=self.settings.retention_lease_seconds,
            now=_utc(self._now()),
        ):
            raise RuntimeError("retention maintenance lease was lost")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as error:  # noqa: BLE001 - keep the daemon alive
                logger.warning(
                    "Retention worker will retry after a failed cleanup cycle",
                    extra={
                        "event": "retention_cleanup_retry_scheduled",
                        "error_type": type(error).__name__,
                        "retry_after_seconds": self.settings.retention_interval_seconds,
                    },
                )
            self._stop_event.wait(self.settings.retention_interval_seconds)


def _snapshot_candidates(
    root: Path,
    *,
    protected_root: Path,
    cutoff: datetime,
    limit: int,
) -> tuple[tuple[Path, int], ...]:
    snapshot_root = root.expanduser().resolve()
    identity_root = protected_root.expanduser().resolve()
    if not snapshot_root.exists():
        return ()

    cutoff_timestamp = cutoff.timestamp()
    candidates: list[tuple[float, Path, int]] = []
    for path in snapshot_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(identity_root):
            continue
        stat = resolved.stat()
        if stat.st_mtime < cutoff_timestamp:
            candidates.append((stat.st_mtime, resolved, stat.st_size))
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return tuple((path, size) for _, path, size in candidates[:limit])


def _snapshot_counts(candidates: tuple[tuple[Path, int], ...]) -> RetentionCounts:
    return RetentionCounts(
        snapshots=len(candidates),
        snapshot_bytes=sum(size for _, size in candidates),
    )


def _in_maintenance_window(
    now: datetime,
    *,
    start_hour_utc: int,
    duration_minutes: int,
) -> bool:
    normalized = _utc(now)
    minute = normalized.hour * 60 + normalized.minute
    start = start_hour_utc * 60
    if duration_minutes >= 1_440:
        return True
    end = start + duration_minutes
    if end <= 1_440:
        return start <= minute < end
    return minute >= start or minute < end % 1_440


def _backup_filename(now: datetime, *, owner_id: str) -> str:
    timestamp = _utc(now).strftime("%Y%m%dT%H%M%S%fZ")
    safe_owner = "".join(character for character in owner_id if character.isalnum())[:16]
    if not safe_owner:
        safe_owner = "worker"
    return f"pre-vacuum-{timestamp}-{safe_owner}.sqlite3"


def _disk_free_bytes(path: Path) -> int:
    return int(disk_usage(path).free)


def _safe_metrics(repository: RetentionRepository) -> RetentionDatabaseMetrics | None:
    try:
        return repository.metrics()
    except Exception:
        logger.exception(
            "Retention database metrics could not be recorded",
            extra={"event": "retention_metrics_failed"},
        )
        return None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("retention timestamps must include timezone information")
    return value.astimezone(UTC)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one lease-protected retention cleanup")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="dry_run", action="store_true")
    mode.add_argument("--apply", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=None)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    arguments = build_argument_parser().parse_args(argv)
    configure_logging(
        level=settings.log_level,
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
        service="cctv-retention-worker",
    )
    try:
        initialize_database(settings.database_path)
        result = RetentionWorker(settings).run_once(dry_run=arguments.dry_run)
        print(json.dumps(asdict(result), default=str, ensure_ascii=False, sort_keys=True))
    finally:
        shutdown_logging()


__all__ = [
    "RetentionRunResult",
    "RetentionRunStatus",
    "RetentionWorker",
    "VacuumRunResult",
    "VacuumRunStatus",
    "build_argument_parser",
    "run",
]
