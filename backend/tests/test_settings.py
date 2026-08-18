from pathlib import Path

from cctv.core.settings import Settings


def test_settings_load_environment_variables(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "test.db"
    model_path = tmp_path / "models" / "test.onnx"

    monkeypatch.setenv("CCTV_APP_ENV", "test")
    monkeypatch.setenv("CCTV_HOST", "0.0.0.0")
    monkeypatch.setenv("CCTV_PORT", "9000")
    monkeypatch.setenv("CCTV_LOG_LEVEL", "debug")
    monkeypatch.setenv("CCTV_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CCTV_MODEL_PATH", str(model_path))
    monkeypatch.setenv("HIPERWALL_AUTH_MODE", "TOKEN")
    monkeypatch.setenv("HIPERWALL_TOKEN", "test-token")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.database_path == database_path
    assert settings.model_path == model_path
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
