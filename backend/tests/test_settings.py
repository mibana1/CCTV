from pathlib import Path

import pytest
from pydantic import ValidationError

from cctv.core.settings import AiDevice, AppMode, Settings


def test_settings_load_environment_variables(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "test.db"
    model_path = tmp_path / "models" / "test.onnx"
    local_video_path = tmp_path / "samples" / "test.mp4"
    snapshot_dir = tmp_path / "snapshots"
    identity_photo_dir = tmp_path / "identity-images"
    face_detection_model_path = tmp_path / "models" / "yunet.onnx"
    face_embedding_model_path = tmp_path / "models" / "sface.onnx"
    model_classes_path = tmp_path / "models" / "classes.txt"
    log_path = tmp_path / "logs" / "test.jsonl"

    monkeypatch.setenv("CCTV_APP_ENV", "test")
    monkeypatch.setenv("CCTV_APP_MODE", "LIVE")
    monkeypatch.setenv("CCTV_AI_DEVICE", "CUDA")
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
        "CCTV_HIPERWALL_DRY_RUN_ENABLED",
        "CCTV_LOCAL_VIDEO_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_mode is AppMode.DRY_RUN
    assert settings.ai_device is AiDevice.CPU
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
    assert settings.hiperwall_dry_run_enabled is True
    assert settings.local_video_path is None
    assert settings.external_actions_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("app_mode", "staging"),
        ("ai_device", "gpu"),
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
