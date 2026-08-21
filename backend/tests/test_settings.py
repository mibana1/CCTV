from pathlib import Path

import pytest
from pydantic import ValidationError

from cctv.core.settings import AiDevice, AppMode, FramePersistenceMode, Settings


def test_settings_load_environment_variables(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "test.db"
    model_path = tmp_path / "models" / "test.onnx"
    local_video_path = tmp_path / "samples" / "test.mp4"
    snapshot_dir = tmp_path / "snapshots"
    identity_photo_dir = tmp_path / "identity-images"
    vacuum_backup_dir = tmp_path / "vacuum-backups"
    face_detection_model_path = tmp_path / "models" / "yunet.onnx"
    face_embedding_model_path = tmp_path / "models" / "sface.onnx"
    model_classes_path = tmp_path / "models" / "classes.txt"
    log_path = tmp_path / "logs" / "test.jsonl"

    monkeypatch.setenv("CCTV_APP_ENV", "test")
    monkeypatch.setenv("CCTV_APP_MODE", "LIVE")
    monkeypatch.setenv("CCTV_AI_DEVICE", "CUDA")
    monkeypatch.setenv("CCTV_AI_ALLOW_CPU_FALLBACK", "true")
    monkeypatch.setenv("CCTV_AI_CUDA_DEVICE_ID", "2")
    monkeypatch.setenv("CCTV_AI_CUDA_GPU_MEM_LIMIT_MB", "4096")
    monkeypatch.setenv("CCTV_ANALYSIS_FPS", "5")
    monkeypatch.setenv("CCTV_DETECTOR_ENABLED", "true")
    monkeypatch.setenv("CCTV_DETECTOR_TYPE", "CUSTOM_DETECTOR")
    monkeypatch.setenv("CCTV_DETECTOR_INPUT_SIZE", "512")
    monkeypatch.setenv("CCTV_DETECTOR_CONFIDENCE_THRESHOLD", "0.35")
    monkeypatch.setenv("CCTV_DETECTOR_NMS_THRESHOLD", "0.4")
    monkeypatch.setenv("CCTV_DETECTION_CLASS_NAMES", " Person, CAR, Cat, DOG, person ")
    monkeypatch.setenv("CCTV_YOLO_ENABLED", "true")
    monkeypatch.setenv("CCTV_YOLO_INPUT_SIZE", "320")
    monkeypatch.setenv("CCTV_YOLO_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("CCTV_YOLO_NMS_THRESHOLD", "0.5")
    monkeypatch.setenv("CCTV_TRACKING_ENABLED", "false")
    monkeypatch.setenv("CCTV_TRACKER_CLASS_NAMES", "PERSON, car")
    monkeypatch.setenv("CCTV_TRACKER_IOU_THRESHOLD", "0.35")
    monkeypatch.setenv("CCTV_TRACKER_MAX_MISSED_FRAMES", "6")
    monkeypatch.setenv("CCTV_TRACKER_MAX_IDLE_SECONDS", "4.5")
    monkeypatch.setenv("CCTV_PERSIST_DETECTIONS", "false")
    monkeypatch.setenv("CCTV_FRAME_PERSISTENCE_MODE", "EVENTS")
    monkeypatch.setenv("CCTV_RETENTION_ENABLED", "true")
    monkeypatch.setenv("CCTV_RETENTION_DRY_RUN", "false")
    monkeypatch.setenv("CCTV_RETENTION_FRAME_DAYS", "14")
    monkeypatch.setenv("CCTV_RETENTION_AUDIT_DAYS", "180")
    monkeypatch.setenv("CCTV_RETENTION_SNAPSHOT_DAYS", "45")
    monkeypatch.setenv("CCTV_RETENTION_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("CCTV_RETENTION_BATCH_SIZE", "250")
    monkeypatch.setenv("CCTV_RETENTION_MAX_BATCHES_PER_RUN", "8")
    monkeypatch.setenv("CCTV_RETENTION_LEASE_SECONDS", "120")
    monkeypatch.setenv("CCTV_RETENTION_CHECKPOINT_ENABLED", "true")
    monkeypatch.setenv("CCTV_RETENTION_TRUNCATE_CHECKPOINT_ENABLED", "true")
    monkeypatch.setenv("CCTV_RETENTION_MAINTENANCE_WINDOW_START_HOUR_UTC", "21")
    monkeypatch.setenv("CCTV_RETENTION_MAINTENANCE_WINDOW_DURATION_MINUTES", "90")
    monkeypatch.setenv("CCTV_RETENTION_VACUUM_ENABLED", "true")
    monkeypatch.setenv("CCTV_RETENTION_VACUUM_FREELIST_RATIO_THRESHOLD", "0.4")
    monkeypatch.setenv("CCTV_RETENTION_VACUUM_MIN_FREELIST_PAGES", "5000")
    monkeypatch.setenv("CCTV_RETENTION_VACUUM_FREE_SPACE_MULTIPLIER", "4")
    monkeypatch.setenv("CCTV_RETENTION_VACUUM_BACKUP_DIR", str(vacuum_backup_dir))
    monkeypatch.setenv("CCTV_HIPERWALL_DRY_RUN_ENABLED", "false")
    monkeypatch.setenv("CCTV_HOST", "0.0.0.0")
    monkeypatch.setenv("CCTV_PORT", "9000")
    monkeypatch.setenv("CCTV_LOG_LEVEL", "debug")
    monkeypatch.setenv("CCTV_LOG_PATH", str(log_path))
    monkeypatch.setenv("CCTV_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("CCTV_LOG_BACKUP_COUNT", "2")
    monkeypatch.setenv("CCTV_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CCTV_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CCTV_MODEL_CLASSES_PATH", str(model_classes_path))
    monkeypatch.setenv("CCTV_LOCAL_VIDEO_PATH", str(local_video_path))
    monkeypatch.setenv("CCTV_SNAPSHOT_DIR", str(snapshot_dir))
    monkeypatch.setenv("CCTV_SNAPSHOT_JPEG_QUALITY", "95")
    monkeypatch.setenv("CCTV_IDENTITY_PHOTO_DIR", str(identity_photo_dir))
    monkeypatch.setenv("CCTV_FACE_DETECTION_MODEL_PATH", str(face_detection_model_path))
    monkeypatch.setenv("CCTV_FACE_EMBEDDING_MODEL_PATH", str(face_embedding_model_path))
    monkeypatch.setenv("CCTV_FACE_DETECTION_SCORE_THRESHOLD", "0.8")
    monkeypatch.setenv("CCTV_FACE_DETECTION_NMS_THRESHOLD", "0.2")
    monkeypatch.setenv("CCTV_FACE_DETECTION_TOP_K", "1000")
    monkeypatch.setenv("CCTV_FACE_DETECTION_MAX_INPUT_DIMENSION", "1280")
    monkeypatch.setenv("CCTV_FACE_MATCHING_ENABLED", "true")
    monkeypatch.setenv("CCTV_FACE_MATCH_SIMILARITY_THRESHOLD", "0.5")
    monkeypatch.setenv("CCTV_FACE_MATCH_MINIMUM_MARGIN", "0.08")
    monkeypatch.setenv("CCTV_FACE_MATCH_UNKNOWN_RETRY_SECONDS", "3.5")
    monkeypatch.setenv("CCTV_FACE_IDENTITY_STITCH_MAX_GAP_SECONDS", "8")
    monkeypatch.setenv("CCTV_FACE_IDENTITY_STITCH_MIN_SIMILARITY", "0.38")
    monkeypatch.setenv("CCTV_FACE_IDENTITY_STITCH_MAX_DISTANCE_RATIO", "7")
    monkeypatch.setenv("CCTV_RTSP_INPUT_URL", "rtsp://user:password@camera:554/stream")
    monkeypatch.setenv("CCTV_RTSP_WORKER_URL", "rtsp://mediamtx:8554/camera")
    monkeypatch.setenv("CCTV_RTSP_SOURCE_NAME", "camera-1")
    monkeypatch.setenv("CCTV_RTSP_OPEN_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("CCTV_RTSP_READ_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("CCTV_RTSP_RECONNECT_INITIAL_SECONDS", "2")
    monkeypatch.setenv("CCTV_RTSP_RECONNECT_MAX_SECONDS", "8")
    monkeypatch.setenv("CCTV_RTSP_RECONNECT_JITTER_RATIO", "0.1")
    monkeypatch.setenv("CCTV_RTSP_MAX_RETRIES", "7")
    monkeypatch.setenv("CCTV_RTSP_MAX_SAMPLES", "9")
    monkeypatch.setenv("HIPERWALL_AUTH_MODE", "TOKEN")
    monkeypatch.setenv("HIPERWALL_USER", "cctv_bridge")
    monkeypatch.setenv("HIPERWALL_TOKEN", "test-token")
    monkeypatch.setenv("HIPERWALL_BASE_URL", "http://hiperwall-host:8000")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.app_mode is AppMode.LIVE
    assert settings.ai_device is AiDevice.CUDA
    assert settings.ai_allow_cpu_fallback is True
    assert settings.ai_cuda_device_id == 2
    assert settings.ai_cuda_gpu_mem_limit_mb == 4096
    assert settings.detector_runtime_options == {
        "allow_cpu_fallback": True,
        "cuda_device_id": 2,
        "cuda_gpu_mem_limit_mb": 4096,
    }
    assert settings.analysis_fps == 5
    assert settings.detector_enabled is True
    assert settings.detector_type == "custom_detector"
    assert settings.effective_detector_input_size == 512
    assert settings.effective_detector_confidence_threshold == 0.35
    assert settings.effective_detector_nms_threshold == 0.4
    assert settings.detection_class_name_list == ("person", "car", "cat", "dog")
    assert settings.object_detection_enabled is True
    assert settings.yolo_enabled is True
    assert settings.yolo_input_size == 320
    assert settings.yolo_confidence_threshold == 0.4
    assert settings.yolo_nms_threshold == 0.5
    assert settings.tracking_enabled is False
    assert settings.tracker_class_name_list == ("person", "car")
    assert settings.tracker_iou_threshold == 0.35
    assert settings.tracker_max_missed_frames == 6
    assert settings.tracker_max_idle_seconds == 4.5
    assert settings.persist_detections is False
    assert settings.frame_persistence_mode is FramePersistenceMode.EVENTS
    assert settings.retention_enabled is True
    assert settings.retention_dry_run is False
    assert settings.retention_frame_days == 14
    assert settings.retention_audit_days == 180
    assert settings.retention_snapshot_days == 45
    assert settings.retention_interval_seconds == 3_600
    assert settings.retention_batch_size == 250
    assert settings.retention_max_batches_per_run == 8
    assert settings.retention_lease_seconds == 120
    assert settings.retention_checkpoint_enabled is True
    assert settings.retention_truncate_checkpoint_enabled is True
    assert settings.retention_maintenance_window_start_hour_utc == 21
    assert settings.retention_maintenance_window_duration_minutes == 90
    assert settings.retention_vacuum_enabled is True
    assert settings.retention_vacuum_freelist_ratio_threshold == 0.4
    assert settings.retention_vacuum_min_freelist_pages == 5_000
    assert settings.retention_vacuum_free_space_multiplier == 4
    assert settings.retention_vacuum_backup_dir == vacuum_backup_dir
    assert settings.hiperwall_dry_run_enabled is False
    assert settings.external_actions_enabled is True
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.log_path == log_path
    assert settings.log_max_bytes == 4_096
    assert settings.log_backup_count == 2
    assert settings.database_path == database_path
    assert settings.model_path == model_path
    assert settings.model_classes_path == model_classes_path
    assert settings.local_video_path == local_video_path
    assert settings.snapshot_dir == snapshot_dir
    assert settings.snapshot_jpeg_quality == 95
    assert settings.identity_photo_dir == identity_photo_dir
    assert settings.face_detection_model_path == face_detection_model_path
    assert settings.face_embedding_model_path == face_embedding_model_path
    assert settings.face_detection_score_threshold == 0.8
    assert settings.face_detection_nms_threshold == 0.2
    assert settings.face_detection_top_k == 1_000
    assert settings.face_detection_max_input_dimension == 1_280
    assert settings.face_matching_enabled is True
    assert settings.face_match_similarity_threshold == 0.5
    assert settings.face_match_minimum_margin == 0.08
    assert settings.face_match_unknown_retry_seconds == 3.5
    assert settings.face_identity_stitch_max_gap_seconds == 8
    assert settings.face_identity_stitch_min_similarity == 0.38
    assert settings.face_identity_stitch_max_distance_ratio == 7
    assert settings.rtsp_input_url == "rtsp://user:password@camera:554/stream"
    assert settings.rtsp_worker_url == "rtsp://mediamtx:8554/camera"
    assert settings.rtsp_source_name == "camera-1"
    assert settings.rtsp_open_timeout_seconds == 4
    assert settings.rtsp_read_timeout_seconds == 5
    assert settings.rtsp_reconnect_initial_seconds == 2
    assert settings.rtsp_reconnect_max_seconds == 8
    assert settings.rtsp_reconnect_jitter_ratio == 0.1
    assert settings.rtsp_max_retries == 7
    assert settings.rtsp_max_samples == 9
    assert settings.hiperwall_auth_mode == "token"
    assert settings.hiperwall_user == "cctv_bridge"
    assert settings.hiperwall_token is not None
    assert settings.hiperwall_token.get_secret_value() == "test-token"


def test_settings_load_explicit_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    database_path = tmp_path / "from-env-file.db"
    env_file.write_text(
        f"CCTV_APP_ENV=env-file-test\nCCTV_DATABASE_PATH={database_path}\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.app_env == "env-file-test"
    assert settings.database_path == database_path


def test_empty_cuda_memory_limit_is_treated_as_unset() -> None:
    settings = Settings(_env_file=None, ai_cuda_gpu_mem_limit_mb="")

    assert settings.ai_cuda_gpu_mem_limit_mb is None
    assert "cuda_gpu_mem_limit_mb" not in settings.detector_runtime_options


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ValidationError, match="log level must be"):
        Settings(_env_file=None, log_level="verbose")


def test_settings_rejects_tracker_classes_outside_detection_filter() -> None:
    with pytest.raises(ValidationError, match="must be a subset"):
        Settings(
            _env_file=None,
            detection_class_names="person,cat",
            tracker_class_names="person,car",
        )


@pytest.mark.parametrize("value", ["", "person,,car", "person,car\tpet"])
def test_settings_rejects_invalid_class_name_lists(value: str) -> None:
    with pytest.raises(ValidationError, match="class names"):
        Settings(_env_file=None, detection_class_names=value)


def test_settings_execution_defaults_are_fail_safe(monkeypatch) -> None:
    for variable in (
        "CCTV_APP_MODE",
        "CCTV_AI_DEVICE",
        "CCTV_AI_ALLOW_CPU_FALLBACK",
        "CCTV_AI_CUDA_DEVICE_ID",
        "CCTV_AI_CUDA_GPU_MEM_LIMIT_MB",
        "CCTV_ANALYSIS_FPS",
        "CCTV_DETECTOR_ENABLED",
        "CCTV_DETECTOR_TYPE",
        "CCTV_DETECTOR_INPUT_SIZE",
        "CCTV_DETECTOR_CONFIDENCE_THRESHOLD",
        "CCTV_DETECTOR_NMS_THRESHOLD",
        "CCTV_DETECTION_CLASS_NAMES",
        "CCTV_YOLO_ENABLED",
        "CCTV_FACE_MATCHING_ENABLED",
        "CCTV_FACE_IDENTITY_STITCH_MAX_GAP_SECONDS",
        "CCTV_FACE_IDENTITY_STITCH_MIN_SIMILARITY",
        "CCTV_FACE_IDENTITY_STITCH_MAX_DISTANCE_RATIO",
        "CCTV_TRACKING_ENABLED",
        "CCTV_TRACKER_CLASS_NAMES",
        "CCTV_PERSIST_DETECTIONS",
        "CCTV_FRAME_PERSISTENCE_MODE",
        "CCTV_RETENTION_ENABLED",
        "CCTV_RETENTION_DRY_RUN",
        "CCTV_RETENTION_FRAME_DAYS",
        "CCTV_RETENTION_AUDIT_DAYS",
        "CCTV_RETENTION_SNAPSHOT_DAYS",
        "CCTV_RETENTION_INTERVAL_SECONDS",
        "CCTV_RETENTION_BATCH_SIZE",
        "CCTV_RETENTION_MAX_BATCHES_PER_RUN",
        "CCTV_RETENTION_LEASE_SECONDS",
        "CCTV_RETENTION_CHECKPOINT_ENABLED",
        "CCTV_RETENTION_TRUNCATE_CHECKPOINT_ENABLED",
        "CCTV_RETENTION_MAINTENANCE_WINDOW_START_HOUR_UTC",
        "CCTV_RETENTION_MAINTENANCE_WINDOW_DURATION_MINUTES",
        "CCTV_RETENTION_VACUUM_ENABLED",
        "CCTV_RETENTION_VACUUM_FREELIST_RATIO_THRESHOLD",
        "CCTV_RETENTION_VACUUM_MIN_FREELIST_PAGES",
        "CCTV_RETENTION_VACUUM_FREE_SPACE_MULTIPLIER",
        "CCTV_RETENTION_VACUUM_BACKUP_DIR",
        "CCTV_HIPERWALL_DRY_RUN_ENABLED",
        "CCTV_LOCAL_VIDEO_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_mode is AppMode.DRY_RUN
    assert settings.ai_device is AiDevice.CPU
    assert settings.ai_allow_cpu_fallback is False
    assert settings.ai_cuda_device_id == 0
    assert settings.ai_cuda_gpu_mem_limit_mb is None
    assert settings.detector_runtime_options == {
        "allow_cpu_fallback": False,
        "cuda_device_id": 0,
    }
    assert settings.analysis_fps == 2
    assert settings.detector_enabled is False
    assert settings.detector_type == "yolo_onnx"
    assert settings.effective_detector_input_size == 640
    assert settings.effective_detector_confidence_threshold == 0.25
    assert settings.effective_detector_nms_threshold == 0.45
    assert settings.detection_class_name_list == ("person", "car", "cat", "dog")
    assert settings.object_detection_enabled is False
    assert settings.yolo_enabled is False
    assert settings.yolo_input_size == 640
    assert settings.yolo_confidence_threshold == 0.25
    assert settings.yolo_nms_threshold == 0.45
    assert settings.face_matching_enabled is False
    assert settings.face_match_similarity_threshold == 0.45
    assert settings.face_match_minimum_margin == 0.05
    assert settings.face_match_unknown_retry_seconds == 2
    assert settings.face_identity_stitch_max_gap_seconds == 12
    assert settings.face_identity_stitch_min_similarity == 0.35
    assert settings.face_identity_stitch_max_distance_ratio == 6
    assert settings.tracking_enabled is True
    assert settings.tracker_class_name_list == ("person",)
    assert settings.tracker_iou_threshold == 0.3
    assert settings.tracker_max_missed_frames == 10
    assert settings.tracker_max_idle_seconds == 12
    assert settings.persist_detections is True
    assert settings.frame_persistence_mode is FramePersistenceMode.ALL
    assert settings.retention_enabled is False
    assert settings.retention_dry_run is True
    assert settings.retention_frame_days == 7
    assert settings.retention_audit_days == 90
    assert settings.retention_snapshot_days == 30
    assert settings.retention_interval_seconds == 86_400
    assert settings.retention_batch_size == 500
    assert settings.retention_max_batches_per_run == 20
    assert settings.retention_lease_seconds == 300
    assert settings.retention_checkpoint_enabled is True
    assert settings.retention_truncate_checkpoint_enabled is False
    assert settings.retention_maintenance_window_start_hour_utc == 18
    assert settings.retention_maintenance_window_duration_minutes == 60
    assert settings.retention_vacuum_enabled is False
    assert settings.retention_vacuum_freelist_ratio_threshold == 0.25
    assert settings.retention_vacuum_min_freelist_pages == 10_000
    assert settings.retention_vacuum_free_space_multiplier == 3
    assert settings.retention_vacuum_backup_dir == Path("runtime/backups/pre-vacuum")
    assert settings.hiperwall_dry_run_enabled is True
    assert settings.local_video_path is None
    assert settings.external_actions_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("app_mode", "staging"),
        ("ai_device", "gpu"),
        ("ai_cuda_device_id", -1),
        ("ai_cuda_gpu_mem_limit_mb", 255),
        ("analysis_fps", 0),
        ("analysis_fps", 31),
        ("detector_type", "invalid detector"),
        ("detector_input_size", 31),
        ("detector_confidence_threshold", 0),
        ("detector_nms_threshold", 1.1),
        ("yolo_input_size", 31),
        ("yolo_input_size", 4097),
        ("yolo_confidence_threshold", 0),
        ("yolo_confidence_threshold", 1.1),
        ("yolo_nms_threshold", -0.1),
        ("yolo_nms_threshold", 1.1),
        ("tracker_iou_threshold", 0),
        ("tracker_iou_threshold", 1.1),
        ("tracker_max_missed_frames", -1),
        ("tracker_max_idle_seconds", 0),
        ("frame_persistence_mode", "sampled"),
        ("retention_frame_days", 0),
        ("retention_audit_days", 0),
        ("retention_snapshot_days", 0),
        ("retention_interval_seconds", 59),
        ("retention_batch_size", 0),
        ("retention_batch_size", 1_001),
        ("retention_max_batches_per_run", 0),
        ("retention_lease_seconds", 29),
        ("retention_maintenance_window_start_hour_utc", -1),
        ("retention_maintenance_window_start_hour_utc", 24),
        ("retention_maintenance_window_duration_minutes", 0),
        ("retention_maintenance_window_duration_minutes", 1_441),
        ("retention_vacuum_freelist_ratio_threshold", 0.04),
        ("retention_vacuum_freelist_ratio_threshold", 0.96),
        ("retention_vacuum_min_freelist_pages", 0),
        ("retention_vacuum_free_space_multiplier", 1.9),
        ("retention_vacuum_free_space_multiplier", 10.1),
        ("snapshot_jpeg_quality", 0),
        ("snapshot_jpeg_quality", 101),
        ("face_detection_score_threshold", 0),
        ("face_detection_nms_threshold", 1.1),
        ("face_detection_top_k", 0),
        ("face_detection_max_input_dimension", 319),
        ("face_match_similarity_threshold", 0),
        ("face_match_minimum_margin", 1.1),
        ("face_match_unknown_retry_seconds", 0),
    ],
)
def test_settings_reject_invalid_execution_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_settings_reject_audit_retention_shorter_than_frame_retention() -> None:
    with pytest.raises(ValueError, match="RETENTION_AUDIT_DAYS"):
        Settings(
            _env_file=None,
            retention_frame_days=30,
            retention_audit_days=7,
        )


@pytest.mark.parametrize(
    "dependent_setting",
    ["retention_truncate_checkpoint_enabled", "retention_vacuum_enabled"],
)
def test_settings_require_checkpoint_for_destructive_maintenance(
    dependent_setting: str,
) -> None:
    with pytest.raises(ValueError, match="RETENTION_CHECKPOINT_ENABLED"):
        Settings(
            _env_file=None,
            retention_checkpoint_enabled=False,
            **{dependent_setting: True},
        )


@pytest.mark.parametrize(
    ("mode", "has_detections", "has_rule_events", "has_display_actions", "expected"),
    [
        (FramePersistenceMode.ALL, False, False, False, True),
        (FramePersistenceMode.DETECTIONS, False, False, False, False),
        (FramePersistenceMode.DETECTIONS, True, False, False, True),
        (FramePersistenceMode.DETECTIONS, False, True, False, True),
        (FramePersistenceMode.EVENTS, True, False, False, False),
        (FramePersistenceMode.EVENTS, False, True, False, True),
        (FramePersistenceMode.EVENTS, False, False, True, True),
    ],
)
def test_frame_persistence_mode_keeps_event_graph_atomic(
    mode: FramePersistenceMode,
    has_detections: bool,
    has_rule_events: bool,
    has_display_actions: bool,
    expected: bool,
) -> None:
    assert (
        mode.should_persist(
            has_detections=has_detections,
            has_rule_events=has_rule_events,
            has_display_actions=has_display_actions,
        )
        is expected
    )


def test_settings_live_mode_requires_hyperwall_url() -> None:
    with pytest.raises(ValidationError, match="HIPERWALL_BASE_URL is required"):
        Settings(
            _env_file=None,
            app_mode="live",
            hiperwall_base_url=None,
        )


def test_settings_live_token_mode_requires_token() -> None:
    with pytest.raises(ValidationError, match="HIPERWALL_TOKEN is required"):
        Settings(
            _env_file=None,
            app_mode="live",
            hiperwall_base_url="http://hiperwall-host:8000",
            hiperwall_auth_mode="token",
            hiperwall_user="cctv_bridge",
            hiperwall_token=None,
        )


def test_settings_live_token_mode_requires_user() -> None:
    with pytest.raises(ValidationError, match="HIPERWALL_USER is required"):
        Settings(
            _env_file=None,
            app_mode="live",
            hiperwall_base_url="http://hiperwall-host:8000",
            hiperwall_auth_mode="token",
            hiperwall_user="",
            hiperwall_token="test-token",
        )


def test_settings_reject_invalid_rtsp_url() -> None:
    with pytest.raises(ValidationError, match="RTSP URL"):
        Settings(_env_file=None, rtsp_worker_url="http://camera/stream")


def test_settings_reject_invalid_analysis_supervisor_restart_window() -> None:
    with pytest.raises(ValueError, match="ANALYSIS_SUPERVISOR_RESTART_MAX_SECONDS"):
        Settings(
            _env_file=None,
            analysis_supervisor_restart_base_seconds=10,
            analysis_supervisor_restart_max_seconds=5,
        )


def test_settings_reject_short_analysis_supervisor_lease() -> None:
    with pytest.raises(ValueError, match="ANALYSIS_SUPERVISOR_LEASE_SECONDS"):
        Settings(
            _env_file=None,
            analysis_supervisor_enabled=True,
            analysis_supervisor_reconcile_interval_seconds=5,
            analysis_supervisor_lease_seconds=10,
        )


@pytest.mark.parametrize(
    "value",
    (
        "rtsp://127.0.0.1:8554",
        "http://user:password@127.0.0.1:18888",
        "http://127.0.0.1:18888?token=secret",
    ),
)
def test_settings_reject_invalid_media_hls_base_url(value: str) -> None:
    with pytest.raises(ValidationError, match="MediaMTX URL"):
        Settings(_env_file=None, media_hls_base_url=value)


def test_settings_reject_reconnect_max_below_initial() -> None:
    with pytest.raises(ValidationError, match="RECONNECT_MAX_SECONDS"):
        Settings(
            _env_file=None,
            rtsp_reconnect_initial_seconds=10,
            rtsp_reconnect_max_seconds=5,
        )
