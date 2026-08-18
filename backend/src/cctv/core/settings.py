"""Application settings loaded from environment variables and the project .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = _BACKEND_ROOT.parent if _BACKEND_ROOT.name.casefold() == "backend" else _BACKEND_ROOT
ENV_FILE = PROJECT_ROOT / ".env"


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
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_path: Path = Path("runtime/logs/cctv.jsonl")
    log_max_bytes: int = Field(default=10_485_760, ge=1_024)
    log_backup_count: int = Field(default=5, ge=0, le=100)

    database_path: Path = Path("runtime/cctv.db")
    model_path: Path = Path("artifacts/models/model.onnx")
    rtsp_input_url: str | None = None
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

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Store log levels in the form expected by logging configuration."""
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")
        return normalized

    @field_validator("hiperwall_auth_mode", mode="before")
    @classmethod
    def normalize_auth_mode(cls, value: object) -> object:
        """Accept case-insensitive authentication mode values."""
        return value.lower() if isinstance(value, str) else value

    @field_validator("database_path", "model_path", "log_path")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        """Resolve relative runtime paths from the repository root."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings instance per application process."""
    return Settings()
