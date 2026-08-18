from pathlib import Path

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.main import create_app


def test_health_initializes_database(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "cctv.db"
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=database_path,
        model_path=tmp_path / "model.onnx",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert database_path.is_file()
    assert client.app.state.database.schema_version == 1
