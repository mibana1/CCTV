"""Application settings loaded from environment variables and the project .env file."""

import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = _BACKEND_ROOT.parent if _BACKEND_ROOT.name.casefold() == "backend" else _BACKEND_ROOT
ENV_FILE = PROJECT_ROOT / ".env"


class AppMode(StrEnum):
    """Whether external side effects are simulated or executed."""

    DRY_RUN = "dry_run"
    LIVE = "live"


class AiDevice(StrEnum):
    """Inference device requested by the analysis worker."""

    CPU = "cpu"
    CUDA = "cuda"


class Settings(BaseSettings):
    """Validated runtime configuration.

    CCTV settings use the ``CCTV_`` prefix. Hiperwall settings retain their
    existing ``HIPERWALL_`` names so the public environment contract remains
    compatible with ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CCTV_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_env: str = "development"
    app_mode: AppMode = AppMode.DRY_RUN
    ai_device: AiDevice = AiDevice.CPU
    analysis_fps: float = Field(default=2.0, gt=0, le=30)
    detector_enabled: bool = False
    detector_type: str = "yolo_onnx"
    detector_input_size: int | None = Field(default=None, ge=32, le=4096)
    detector_confidence_threshold: float | None = Field(default=None, gt=0, le=1)
    detector_nms_threshold: float | None = Field(default=None, ge=0, le=1)
    detection_class_names: str = "person,car,cat,dog"
    # Deprecated compatibility settings. Generic detector settings take precedence.
    yolo_enabled: bool = False
    yolo_input_size: int = Field(default=640, ge=32, le=4096)
    yolo_confidence_threshold: float = Field(default=0.25, gt=0, le=1)
    yolo_nms_threshold: float = Field(default=0.45, ge=0, le=1)
    tracking_enabled: bool = True
    tracker_iou_threshold: float = Field(default=0.3, gt=0, le=1)
    tracker_max_missed_frames: int = Field(default=10, ge=0, le=300)
    tracker_max_idle_seconds: float = Field(default=12.0, gt=0, le=300)
    tracker_class_names: str = "person"
    persist_detections: bool = True
    rules_enabled: bool = True
    hiperwall_dry_run_enabled: bool = True
    hiperwall_executor_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_path: Path = Path("runtime/logs/cctv.jsonl")
    log_max_bytes: int = Field(default=10_485_760, ge=1_024)
    log_backup_count: int = Field(default=5, ge=0, le=100)
    snapshot_dir: Path = Path("runtime/snapshots")
    snapshot_jpeg_quality: int = Field(default=90, ge=1, le=100)
    identity_photo_dir: Path = Path("runtime/identity_images")

    database_path: Path = Path("runtime/cctv.db")
    model_path: Path = Path("artifacts/models/model.onnx")
    face_detection_model_path: Path = Path(
        "artifacts/models/face/face_detection_yunet_2026may.onnx"
    )
    face_embedding_model_path: Path = Path(
        "artifacts/models/face/face_recognition_sface_2021dec.onnx"
    )
    face_detection_score_threshold: float = Field(default=0.8, gt=0, le=1)
    face_detection_nms_threshold: float = Field(default=0.3, ge=0, le=1)
    face_detection_top_k: int = Field(default=5_000, ge=1, le=100_000)
    face_detection_max_input_dimension: int = Field(default=960, ge=320, le=4_096)
    face_matching_enabled: bool = False
    face_match_similarity_threshold: float = Field(default=0.45, gt=0, le=1)
    face_match_minimum_margin: float = Field(default=0.05, ge=0, le=1)
    face_match_unknown_retry_seconds: float = Field(default=2.0, gt=0, le=300)
    face_identity_stitch_max_gap_seconds: float = Field(default=12.0, gt=0, le=300)
    face_identity_stitch_min_similarity: float = Field(default=0.35, ge=-1, le=1)
    face_identity_stitch_max_distance_ratio: float = Field(default=6.0, gt=0, le=100)
    test_dashboard_enabled: bool = False
    media_hls_base_url: str = "http://127.0.0.1:18888"
    mediamtx_dynamic_paths_enabled: bool = False
    mediamtx_api_url: str = "http://127.0.0.1:9997"
    mediamtx_api_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    mediamtx_reconcile_interval_seconds: float = Field(default=10.0, ge=2, le=300)
    camera_credential_key: SecretStr | None = None
    camera_credential_key_path: Path = Path("runtime/secrets/camera_credentials.key")
    model_classes_path: Path | None = None
    local_video_path: Path | None = None
    rtsp_input_url: str | None = None
    rtsp_worker_url: str = "rtsp://127.0.0.1:8554/camera"
    rtsp_source_name: str = "camera"
    rtsp_open_timeout_seconds: float = Field(default=10, gt=0, le=120)
    rtsp_read_timeout_seconds: float = Field(default=10, gt=0, le=120)
    rtsp_reconnect_initial_seconds: float = Field(default=1, gt=0, le=60)
    rtsp_reconnect_max_seconds: float = Field(default=30, gt=0, le=300)
    rtsp_reconnect_jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    rtsp_max_retries: int = Field(default=20, ge=0, le=10_000)
    rtsp_max_samples: int | None = Field(default=None, ge=1)
    restream_url: str = "rtsp://127.0.0.1:8554/analyzed"

    hiperwall_base_url: str | None = Field(
        default=None,
        validation_alias="HIPERWALL_BASE_URL",
    )
    hiperwall_auth_mode: Literal["none", "token", "crypto"] = Field(
        default="none",
        validation_alias="HIPERWALL_AUTH_MODE",
    )
    hiperwall_token: SecretStr | None = Field(
        default=None,
        validation_alias="HIPERWALL_TOKEN",
    )
    hiperwall_user: str = Field(
        default="",
        validation_alias="HIPERWALL_USER",
    )
    hiperwall_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        le=60,
        validation_alias="HIPERWALL_TIMEOUT_SECONDS",
    )
    hiperwall_poll_interval_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=60,
        validation_alias="HIPERWALL_POLL_INTERVAL_SECONDS",
    )
    hiperwall_max_attempts: int = Field(
        default=8,
        ge=1,
        le=100,
        validation_alias="HIPERWALL_MAX_ATTEMPTS",
    )
    hiperwall_retry_base_seconds: float = Field(
        default=5.0,
        gt=0,
        le=3_600,
        validation_alias="HIPERWALL_RETRY_BASE_SECONDS",
    )
    hiperwall_retry_max_seconds: float = Field(
        default=300.0,
        gt=0,
        le=86_400,
        validation_alias="HIPERWALL_RETRY_MAX_SECONDS",
    )
    hiperwall_default_display_seconds: int = Field(
        default=30,
        ge=1,
        le=86_400,
        validation_alias="HIPERWALL_DEFAULT_DISPLAY_SECONDS",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Store log levels in the form expected by logging configuration."""
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")
        return normalized

    @field_validator("app_mode", "ai_device", mode="before")
    @classmethod
    def normalize_execution_mode(cls, value: object) -> object:
        """Accept case-insensitive execution mode and device values."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("hiperwall_auth_mode", mode="before")
    @classmethod
    def normalize_auth_mode(cls, value: object) -> object:
        """Accept case-insensitive authentication mode values."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("hiperwall_user")
    @classmethod
    def normalize_hiperwall_user(cls, value: str) -> str:
        return value.strip()

    @field_validator("detector_type")
    @classmethod
    def normalize_detector_type(cls, value: str) -> str:
        """Normalize a registry key while allowing future detector adapters."""
        normalized = value.strip().casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", normalized):
            raise ValueError(
                "detector type must be 1-64 lowercase letters, numbers, underscores, or hyphens"
            )
        return normalized

    @field_validator(
        "database_path",
        "model_path",
        "log_path",
        "snapshot_dir",
        "identity_photo_dir",
        "face_detection_model_path",
        "face_embedding_model_path",
        "camera_credential_key_path",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        """Resolve relative runtime paths from the repository root."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator(
        "local_video_path",
        "model_classes_path",
        "rtsp_input_url",
        "camera_credential_key",
        "hiperwall_base_url",
        mode="before",
    )
    @classmethod
    def normalize_optional_value(cls, value: object) -> object:
        """Treat empty optional environment variables as unset."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("local_video_path", "model_classes_path")
    @classmethod
    def resolve_optional_project_path(cls, value: Path | None) -> Path | None:
        """Resolve optional project-owned file paths from the repository root."""
        if value is None or value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()

    @field_validator("rtsp_input_url", "rtsp_worker_url", "restream_url")
    @classmethod
    def validate_rtsp_url(cls, value: str | None) -> str | None:
        """Require an RTSP(S) URL with a host without exposing its credentials."""
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("RTSP URL must use rtsp:// or rtsps:// and include a host")
        return value

    @field_validator("media_hls_base_url", "mediamtx_api_url")
    @classmethod
    def validate_mediamtx_http_url(cls, value: str) -> str:
        """Require credential-free HTTP endpoints for MediaMTX services."""
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "MediaMTX URL must be a credential-free http(s) URL without query or fragment"
            )
        return normalized

    @field_validator("hiperwall_base_url")
    @classmethod
    def validate_hiperwall_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "HIPERWALL_BASE_URL must be a credential-free http(s) URL without query or fragment"
            )
        return normalized

    @field_validator("detection_class_names", "tracker_class_names")
    @classmethod
    def normalize_class_name_list(cls, value: str) -> str:
        """Normalize a comma-separated, ordered set of detector class labels."""
        raw_names = value.split(",")
        names = tuple(dict.fromkeys(name.strip().casefold() for name in raw_names))
        if not names or any(not name or len(name) > 128 for name in names):
            raise ValueError(
                "class names must be a comma-separated list of 1-128 character labels"
            )
        if any(any(character in "\r\n\t" for character in name) for name in names):
            raise ValueError("class names must not contain control characters")
        return ",".join(names)

    @field_validator("rtsp_source_name")
    @classmethod
    def validate_rtsp_source_name(cls, value: str) -> str:
        """Keep the public source label safe for logs, paths, and database rows."""
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", normalized):
            raise ValueError(
                "RTSP source name must be 1-64 letters, numbers, dots, underscores, or hyphens"
            )
        return normalized

    @model_validator(mode="after")
    def validate_live_mode(self) -> Self:
        """Fail closed when LIVE mode lacks required Hiperwall configuration."""
        if self.rtsp_reconnect_max_seconds < self.rtsp_reconnect_initial_seconds:
            raise ValueError(
                "CCTV_RTSP_RECONNECT_MAX_SECONDS must be greater than or equal to "
                "CCTV_RTSP_RECONNECT_INITIAL_SECONDS"
            )
        unavailable_tracker_classes = (
            self.tracker_class_name_set - self.detection_class_name_set
        )
        if unavailable_tracker_classes:
            formatted = ", ".join(sorted(unavailable_tracker_classes))
            raise ValueError(
                "CCTV_TRACKER_CLASS_NAMES must be a subset of "
                f"CCTV_DETECTION_CLASS_NAMES; unavailable: {formatted}"
            )
        if self.app_mode is not AppMode.LIVE:
            return self
        if not self.hiperwall_executor_enabled:
            return self
        if not self.hiperwall_base_url:
            raise ValueError("HIPERWALL_BASE_URL is required when CCTV_APP_MODE=live")
        if self.hiperwall_auth_mode == "crypto":
            raise ValueError("HIPERWALL_AUTH_MODE=crypto is not supported in live mode")
        if self.hiperwall_auth_mode == "token":
            if not self.hiperwall_user:
                raise ValueError(
                    "HIPERWALL_USER is required for token authentication in live mode"
                )
            if self.hiperwall_token is None or not self.hiperwall_token.get_secret_value():
                raise ValueError(
                    "HIPERWALL_TOKEN is required for token authentication in live mode"
                )
        if self.hiperwall_retry_max_seconds < self.hiperwall_retry_base_seconds:
            raise ValueError(
                "HIPERWALL_RETRY_MAX_SECONDS must be greater than or equal to "
                "HIPERWALL_RETRY_BASE_SECONDS"
            )
        return self

    @property
    def external_actions_enabled(self) -> bool:
        """Report whether external side effects may be executed."""
        return self.app_mode is AppMode.LIVE

    @property
    def object_detection_enabled(self) -> bool:
        """Honor the generic switch while retaining CCTV_YOLO_ENABLED compatibility."""
        return self.detector_enabled or self.yolo_enabled

    @property
    def effective_detector_input_size(self) -> int:
        return self.detector_input_size or self.yolo_input_size

    @property
    def effective_detector_confidence_threshold(self) -> float:
        return self.detector_confidence_threshold or self.yolo_confidence_threshold

    @property
    def effective_detector_nms_threshold(self) -> float:
        if self.detector_nms_threshold is not None:
            return self.detector_nms_threshold
        return self.yolo_nms_threshold

    @property
    def detection_class_name_list(self) -> tuple[str, ...]:
        return tuple(self.detection_class_names.split(","))

    @property
    def detection_class_name_set(self) -> frozenset[str]:
        return frozenset(self.detection_class_name_list)

    @property
    def tracker_class_name_list(self) -> tuple[str, ...]:
        return tuple(self.tracker_class_names.split(","))

    @property
    def tracker_class_name_set(self) -> frozenset[str]:
        return frozenset(self.tracker_class_name_list)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings instance per application process."""
    return Settings()
