import json
import logging
from pathlib import Path

from cctv.core.logging import JsonFormatter, configure_logging, shutdown_logging


def test_json_formatter_adds_common_fields() -> None:
    formatter = JsonFormatter(service="test-service")
    record = logging.LogRecord(
        name="cctv.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="Camera connected",
        args=(),
        exc_info=None,
    )
    record.event = "camera_connected"
    record.camera_id = "cam-001"

    payload = json.loads(formatter.format(record))

    assert payload["timestamp"].endswith("Z")
    assert payload["level"] == "INFO"
    assert payload["service"] == "test-service"
    assert payload["logger"] == "cctv.test"
    assert payload["message"] == "Camera connected"
    assert payload["event"] == "camera_connected"
    assert payload["camera_id"] == "cam-001"


def test_json_formatter_redacts_credentials_recursively() -> None:
    formatter = JsonFormatter(service="test-service")
    record = logging.LogRecord(
        name="cctv.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=30,
        msg=(
            "Connect rtsp://camera-user:camera-password@camera-host:554/stream "
            "with Bearer hiperwall-secret <Token>xml-token</Token> "
            "https://hiperwall-host/control?token=query-token"
        ),
        args=(),
        exc_info=None,
    )
    record.context = {
        "password": "nested-password",
        "hiperwall_token": "nested-token",
        "headers": {"Authorization": "Bearer nested-authorization"},
        "session_cookie": "nested-cookie",
        "rtsp_url": "rtsps://nested-user:nested-rtsp-secret@camera-host/stream",
        "safe_value": "visible",
    }

    rendered = formatter.format(record)
    payload = json.loads(rendered)

    for secret in (
        "camera-user",
        "camera-password",
        "hiperwall-secret",
        "xml-token",
        "query-token",
        "nested-password",
        "nested-token",
        "nested-authorization",
        "nested-cookie",
        "nested-user",
        "nested-rtsp-secret",
    ):
        assert secret not in rendered

    assert "rtsp://***:***@camera-host:554/stream" in payload["message"]
    assert payload["context"]["password"] == "***"
    assert payload["context"]["hiperwall_token"] == "***"
    assert payload["context"]["headers"]["Authorization"] == "***"
    assert payload["context"]["session_cookie"] == "***"
    assert payload["context"]["safe_value"] == "visible"


def test_configure_logging_writes_jsonl_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "cctv.jsonl"
    configure_logging(
        level="INFO",
        log_path=log_path,
        max_bytes=4_096,
        backup_count=2,
        service="test-service",
    )

    try:
        logging.getLogger("cctv.worker").info(
            "Frame sampled",
            extra={"event": "frame_sampled", "camera_id": "cam-001"},
        )
    finally:
        shutdown_logging()

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "frame_sampled"
    assert payload["camera_id"] == "cam-001"
    assert payload["service"] == "test-service"


def test_configure_logging_rotates_jsonl_files(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "cctv.jsonl"
    configure_logging(
        level="INFO",
        log_path=log_path,
        max_bytes=256,
        backup_count=2,
        service="test-service",
    )

    try:
        logger = logging.getLogger("cctv.rotation")
        for index in range(10):
            logger.info(
                "Rotation test payload %s",
                "x" * 100,
                extra={"event": "rotation_test", "record_index": index},
            )
    finally:
        shutdown_logging()

    rotated_files = sorted(log_path.parent.glob("cctv.jsonl*"))
    assert log_path in rotated_files
    assert log_path.with_name("cctv.jsonl.1") in rotated_files
    assert len(rotated_files) <= 3
