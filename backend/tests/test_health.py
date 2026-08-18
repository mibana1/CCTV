import json
from pathlib import Path

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.main import create_app


def test_health_initializes_database(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "cctv.db"
    log_path = tmp_path / "runtime" / "logs" / "cctv.jsonl"
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=database_path,
        model_path=tmp_path / "model.onnx",
        log_path=log_path,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert database_path.is_file()
    assert client.app.state.database.schema_version == 1

    log_records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    events = {record["event"] for record in log_records}
    assert {
        "application_starting",
        "database_migration_applied",
        "database_initialized",
        "application_started",
        "http_request_completed",
        "application_stopped",
    } <= events

    request_record = next(
        record for record in log_records if record["event"] == "http_request_completed"
    )
    assert request_record["http_method"] == "GET"
    assert request_record["http_path"] == "/health"
    assert request_record["status_code"] == 200
    assert request_record["duration_ms"] >= 0
