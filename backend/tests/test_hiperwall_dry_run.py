from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import DetectionRepository, DisplayActionRepository, RuleRepository
from cctv.hiperwall import HiperwallDryRunPlanner
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.main import create_app
from cctv.rules import RuleEvent


def _frame() -> FrameDetections:
    return FrameDetections(
        source_index=3,
        sample_index=1,
        timestamp_seconds=1.5,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.02,
        detections=(
            Detection(
                class_id=0,
                label="person",
                confidence=0.93,
                box=BoundingBox(100, 50, 220, 300),
                track_id=7,
            ),
        ),
    )


def _events(rule_id: str) -> tuple[RuleEvent, RuleEvent]:
    return (
        RuleEvent(
            id="event-started",
            rule_id=rule_id,
            track_id=7,
            event_type="intrusion_started",
            event_state="started",
            occurred_at_seconds=1.5,
            class_name="person",
            confidence=0.93,
            payload={"point": [0.25, 0.83]},
        ),
        RuleEvent(
            id="event-ended",
            rule_id=rule_id,
            track_id=7,
            event_type="intrusion_ended",
            event_state="ended",
            occurred_at_seconds=2.5,
            class_name="person",
            confidence=0.9,
            payload={"reason": "outside"},
        ),
    )


def _seed_actions(database_path: Path, settings: Settings) -> tuple[str, tuple[str, ...]]:
    rules = RuleRepository(database_path)
    rule = rules.create_rule(
        rule_id="rule-1",
        name="Restricted area",
        source_name="camera-1",
        rule_type="intrusion",
        class_name="person",
        geometry={
            "type": "polygon",
            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
    )
    detections = DetectionRepository(database_path)
    run = detections.create_analysis_run(
        source_type="rtsp",
        source_name="camera-1",
        detector_type="test_detector",
        model_name="model.onnx",
        model_sha256="a" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
        run_id="run-1",
    )
    events = _events(rule.id)
    actions = HiperwallDryRunPlanner(settings).plan(
        events,
        analysis_run_id=run.id,
        source_name="camera-1",
    )
    detections.save_frame(
        run.id,
        _frame(),
        active_track_ids={7},
        rule_events=events,
        display_actions=actions,
    )
    return run.id, tuple(action.id for action in actions)


def test_dry_run_planner_maps_events_without_credentials_or_network_request(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_mode="dry_run",
        hiperwall_base_url="http://hiperwall-host:8000",
        hiperwall_token="super-secret-token",
        database_path=tmp_path / "cctv.db",
    )
    events = _events("rule-1")

    actions = HiperwallDryRunPlanner(settings).plan(
        events,
        analysis_run_id="run-1",
        source_name="camera-1",
    )

    assert [action.action_type for action in actions] == ["open_source", "restore_layout"]
    assert all(action.mode == "dry_run" for action in actions)
    assert all(action.status == "simulated" for action in actions)
    assert all(
        action.result == {"external_request_sent": False, "reason": "dry_run"} for action in actions
    )
    serialized_requests = repr([action.request for action in actions])
    assert "super-secret-token" not in serialized_requests
    assert "hiperwall-host" not in serialized_requests


def test_dry_run_planner_rejects_live_mode() -> None:
    settings = Settings(
        _env_file=None,
        app_mode="live",
        hiperwall_base_url="http://hiperwall-host:8000",
    )

    with pytest.raises(ValueError, match="CCTV_APP_MODE=dry_run"):
        HiperwallDryRunPlanner(settings)


def test_display_actions_are_persisted_and_exposed_by_api(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "cctv.db"
    settings = Settings(
        _env_file=None,
        app_env="test",
        app_mode="dry_run",
        database_path=database_path,
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "runtime" / "cctv.jsonl",
    )

    with TestClient(create_app(settings)) as client:
        run_id, action_ids = _seed_actions(database_path, settings)
        response = client.get(
            "/hiperwall-actions",
            params={
                "analysis_run_id": run_id,
                "source_name": "camera-1",
                "status": "simulated",
            },
        )
        open_actions = client.get(
            "/hiperwall-actions",
            params={"action_type": "open_source"},
        )
        detail = client.get(f"/hiperwall-actions/{action_ids[0]}")

        assert response.status_code == 200
        assert response.json()["total"] == 2
        assert {item["action_type"] for item in response.json()["items"]} == {
            "open_source",
            "restore_layout",
        }
        assert all(
            item["result"]["external_request_sent"] is False for item in response.json()["items"]
        )
        assert open_actions.status_code == 200
        assert open_actions.json()["total"] == 1
        assert detail.status_code == 200
        assert detail.json()["rule_event_id"] == "event-started"
        assert detail.json()["request"]["source_name"] == "camera-1"

        repository = DisplayActionRepository(database_path)
        assert repository.list(analysis_run_id=run_id).total == 2
        assert client.get("/hiperwall-actions/missing").status_code == 404
        assert client.get("/hiperwall-actions", params={"limit": 101}).status_code == 422
