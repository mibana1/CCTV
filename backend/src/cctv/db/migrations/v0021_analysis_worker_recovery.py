"""Add interrupted runs, run-bound leases, and durable worker failures."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    _rebuild_analysis_runs(connection)
    connection.execute(
        """
        ALTER TABLE analysis_worker_leases
        ADD COLUMN analysis_run_id TEXT
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_analysis_worker_leases_run
        ON analysis_worker_leases (analysis_run_id)
        WHERE analysis_run_id IS NOT NULL
        """
    )
    connection.execute(
        """
        CREATE TABLE analysis_worker_failures (
            id INTEGER PRIMARY KEY,
            camera_id INTEGER NOT NULL
                REFERENCES cameras(id) ON DELETE CASCADE,
            analysis_run_id TEXT
                REFERENCES analysis_runs(id) ON DELETE SET NULL,
            owner_id TEXT NOT NULL CHECK (length(owner_id) BETWEEN 1 AND 128),
            process_instance_id TEXT NOT NULL UNIQUE
                CHECK (length(process_instance_id) BETWEEN 1 AND 128),
            failure_type TEXT NOT NULL CHECK (length(failure_type) BETWEEN 1 AND 128),
            failure_message TEXT NOT NULL CHECK (length(failure_message) BETWEEN 1 AND 1024),
            failure_at TEXT NOT NULL,
            restart_count INTEGER NOT NULL CHECK (restart_count >= 1),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_analysis_worker_failures_camera_time
        ON analysis_worker_failures (camera_id, failure_at DESC, id DESC)
        """
    )


def _rebuild_analysis_runs(connection: Connection) -> None:
    # This is the latest migration. The connection is not yet in a transaction, so
    # foreign key rewriting can be disabled while the parent table is rebuilt.
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("PRAGMA legacy_alter_table = ON")
    connection.execute("ALTER TABLE analysis_runs RENAME TO analysis_runs_v0020")
    connection.execute(
        """
        CREATE TABLE analysis_runs (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK (source_type IN ('local_video', 'rtsp')),
            source_name TEXT NOT NULL CHECK (length(source_name) > 0),
            camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
            detector_type TEXT NOT NULL DEFAULT 'yolo_onnx'
                CHECK (length(detector_type) > 0),
            model_name TEXT NOT NULL CHECK (length(model_name) > 0),
            model_sha256 TEXT NOT NULL CHECK (length(model_sha256) = 64),
            device TEXT NOT NULL CHECK (device IN ('cpu', 'cuda')),
            input_size INTEGER NOT NULL CHECK (input_size >= 32),
            confidence_threshold REAL NOT NULL
                CHECK (confidence_threshold > 0 AND confidence_threshold <= 1),
            nms_threshold REAL NOT NULL CHECK (nms_threshold >= 0 AND nms_threshold <= 1),
            sample_fps REAL NOT NULL CHECK (sample_fps > 0),
            status TEXT NOT NULL CHECK (
                status IN ('running', 'completed', 'stopped', 'interrupted', 'failed')
            ),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            processed_frames INTEGER NOT NULL DEFAULT 0 CHECK (processed_frames >= 0),
            total_detections INTEGER NOT NULL DEFAULT 0 CHECK (total_detections >= 0),
            error_type TEXT,
            completion_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (
                (status = 'running' AND completed_at IS NULL)
                OR (status != 'running' AND completed_at IS NOT NULL)
            )
        )
        """
    )
    connection.execute(
        """
        INSERT INTO analysis_runs (
            id, source_type, source_name, camera_id, detector_type,
            model_name, model_sha256, device, input_size,
            confidence_threshold, nms_threshold, sample_fps,
            status, started_at, completed_at, processed_frames,
            total_detections, error_type, created_at
        )
        SELECT
            id, source_type, source_name, camera_id, detector_type,
            model_name, model_sha256, device, input_size,
            confidence_threshold, nms_threshold, sample_fps,
            status, started_at, completed_at, processed_frames,
            total_detections, error_type, created_at
        FROM analysis_runs_v0020
        """
    )
    connection.execute("DROP TABLE analysis_runs_v0020")
    connection.execute(
        """
        CREATE INDEX idx_analysis_runs_started
        ON analysis_runs (started_at DESC, id DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_analysis_runs_camera
        ON analysis_runs (camera_id, started_at DESC)
        """
    )
    connection.execute("PRAGMA legacy_alter_table = OFF")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("analysis_runs rebuild produced foreign key violations")


__all__ = ["apply"]
