import json
from pathlib import Path

from cctv.core.execution import external_action_allowed
from cctv.core.logging import configure_logging, shutdown_logging
from cctv.core.settings import Settings


def test_dry_run_blocks_and_logs_external_action(tmp_path: Path) -> None:
    log_path = tmp_path / "cctv.jsonl"
    settings = Settings(_env_file=None, app_mode="dry_run")
    configure_logging(
        level="INFO",
        log_path=log_path,
        max_bytes=4_096,
        backup_count=1,
        service="test-service",
    )

    try:
        allowed = external_action_allowed(settings, action="hiperwall_open")
    finally:
        shutdown_logging()

    assert allowed is False
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["event"] == "external_action_skipped"
    assert record["action"] == "hiperwall_open"
    assert record["reason"] == "dry_run"
    assert record["app_mode"] == "dry_run"


def test_live_mode_allows_external_action() -> None:
    settings = Settings(
        _env_file=None,
        app_mode="live",
        hiperwall_base_url="http://hiperwall-host:8000",
    )

    assert external_action_allowed(settings, action="hiperwall_open") is True
