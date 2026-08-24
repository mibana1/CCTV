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
    assert response.json() == {
        "status": "ok",
        "service": "cctv-backend",
        "version": "0.1.0",
        "environment": "test",
        "mode": "dry_run",
        "checks": {
            "database": {
                "status": "ok",
                "schema_version": 21,
                "journal_mode": "wal",
            }
        },
    }
    assert database_path.is_file()
    assert client.app.state.database.schema_version == 21

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

    startup_record = next(
        record for record in log_records if record["event"] == "application_starting"
    )
    assert startup_record["app_env"] == "test"
    assert startup_record["app_mode"] == "dry_run"
    assert startup_record["ai_device"] == "cpu"
    assert startup_record["analysis_fps"] == 2
    assert startup_record["detector_enabled"] is False
    assert startup_record["detector_type"] == "yolo_onnx"


def test_health_returns_service_unavailable_when_database_is_missing(tmp_path: Path) -> None:
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
        database_path.unlink()
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "error",
        "service": "cctv-backend",
        "version": "0.1.0",
        "environment": "test",
        "mode": "dry_run",
        "checks": {"database": {"status": "error"}},
    }

    log_records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    failure_record = next(
        record for record in log_records if record["event"] == "health_check_failed"
    )
    assert failure_record["component"] == "database"
    assert failure_record["error_type"] == "OperationalError"

    request_record = next(
        record
        for record in log_records
        if record["event"] == "http_request_completed" and record["status_code"] == 503
    )
    assert request_record["http_path"] == "/health"
