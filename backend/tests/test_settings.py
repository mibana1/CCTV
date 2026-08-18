from pathlib import Path

import pytest
from pydantic import ValidationError

from cctv.core.settings import AiDevice, AppMode, Settings


def test_settings_load_environment_variables(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "test.db"
    model_path = tmp_path / "models" / "test.onnx"
    local_video_path = tmp_path / "samples" / "test.mp4"
    log_path = tmp_path / "logs" / "test.jsonl"

    monkeypatch.setenv("CCTV_APP_ENV", "test")
    monkeypatch.setenv("CCTV_APP_MODE", "LIVE")
    monkeypatch.setenv("CCTV_AI_DEVICE", "CUDA")
    monkeypatch.setenv("CCTV_ANALYSIS_FPS", "5")
    monkeypatch.setenv("CCTV_HOST", "0.0.0.0")
    monkeypatch.setenv("CCTV_PORT", "9000")
    monkeypatch.setenv("CCTV_LOG_LEVEL", "debug")
    monkeypatch.setenv("CCTV_LOG_PATH", str(log_path))
    monkeypatch.setenv("CCTV_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("CCTV_LOG_BACKUP_COUNT", "2")
    monkeypatch.setenv("CCTV_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CCTV_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CCTV_LOCAL_VIDEO_PATH", str(local_video_path))
    monkeypatch.setenv("HIPERWALL_AUTH_MODE", "TOKEN")
    monkeypatch.setenv("HIPERWALL_TOKEN", "test-token")
    monkeypatch.setenv("HIPERWALL_BASE_URL", "http://hiperwall-host:8000")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.app_mode is AppMode.LIVE
    assert settings.ai_device is AiDevice.CUDA
    assert settings.analysis_fps == 5
    assert settings.external_actions_enabled is True
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.log_path == log_path
    assert settings.log_max_bytes == 4_096
    assert settings.log_backup_count == 2
    assert settings.database_path == database_path
    assert settings.model_path == model_path
    assert settings.local_video_path == local_video_path
    assert settings.hiperwall_auth_mode == "token"
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


def test_settings_execution_defaults_are_fail_safe(monkeypatch) -> None:
    for variable in (
        "CCTV_APP_MODE",
        "CCTV_AI_DEVICE",
        "CCTV_ANALYSIS_FPS",
        "CCTV_LOCAL_VIDEO_PATH",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_mode is AppMode.DRY_RUN
    assert settings.ai_device is AiDevice.CPU
    assert settings.analysis_fps == 2
    assert settings.local_video_path is None
    assert settings.external_actions_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("app_mode", "staging"),
        ("ai_device", "gpu"),
        ("analysis_fps", 0),
        ("analysis_fps", 31),
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
            hiperwall_token=None,
        )
