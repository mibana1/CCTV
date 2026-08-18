"""Create analysis-run, analyzed-frame, and object-detection tables."""

from sqlite3 import Connection


def apply(connection: Connection) -> None:
    """Create normalized detection-result storage and query indexes."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK (source_type IN ('local_video', 'rtsp')),
            source_name TEXT NOT NULL CHECK (length(source_name) > 0),
            camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
            model_name TEXT NOT NULL CHECK (length(model_name) > 0),
            model_sha256 TEXT NOT NULL CHECK (length(model_sha256) = 64),
            device TEXT NOT NULL CHECK (device IN ('cpu', 'cuda')),
            input_size INTEGER NOT NULL CHECK (input_size >= 32),
            confidence_threshold REAL NOT NULL
                CHECK (confidence_threshold > 0 AND confidence_threshold <= 1),
            nms_threshold REAL NOT NULL CHECK (nms_threshold >= 0 AND nms_threshold <= 1),
            sample_fps REAL NOT NULL CHECK (sample_fps > 0),
            status TEXT NOT NULL
                CHECK (status IN ('running', 'completed', 'stopped', 'failed')),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            processed_frames INTEGER NOT NULL DEFAULT 0 CHECK (processed_frames >= 0),
            total_detections INTEGER NOT NULL DEFAULT 0 CHECK (total_detections >= 0),
            error_type TEXT,
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
        CREATE TABLE IF NOT EXISTS analyzed_frames (
            id INTEGER PRIMARY KEY,
            analysis_run_id TEXT NOT NULL
                REFERENCES analysis_runs(id) ON DELETE CASCADE,
            source_index INTEGER NOT NULL CHECK (source_index >= 0),
            sample_index INTEGER NOT NULL CHECK (sample_index >= 0),
            source_timestamp_seconds REAL NOT NULL CHECK (source_timestamp_seconds >= 0),
            frame_width INTEGER NOT NULL CHECK (frame_width > 0),
            frame_height INTEGER NOT NULL CHECK (frame_height > 0),
            inference_seconds REAL NOT NULL CHECK (inference_seconds >= 0),
            detection_count INTEGER NOT NULL CHECK (detection_count >= 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (analysis_run_id, sample_index)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            analyzed_frame_id INTEGER NOT NULL
                REFERENCES analyzed_frames(id) ON DELETE CASCADE,
            class_id INTEGER NOT NULL CHECK (class_id >= 0),
            class_name TEXT NOT NULL CHECK (length(class_name) > 0),
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            x1 INTEGER NOT NULL CHECK (x1 >= 0),
            y1 INTEGER NOT NULL CHECK (y1 >= 0),
            x2 INTEGER NOT NULL CHECK (x2 > x1),
            y2 INTEGER NOT NULL CHECK (y2 > y1),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_runs_started
        ON analysis_runs (started_at DESC, id DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_runs_camera
        ON analysis_runs (camera_id, started_at DESC)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analyzed_frames_run_timestamp
        ON analyzed_frames (analysis_run_id, source_timestamp_seconds, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_frame
        ON detections (analyzed_frame_id, id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_class_confidence
        ON detections (class_name, confidence DESC, id)
        """
    )
