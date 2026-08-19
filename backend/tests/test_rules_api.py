from pathlib import Path

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import DetectionRepository
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.main import create_app
from cctv.rules import RuleEvent


def test_rules_api_manages_definitions_and_queries_events(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "cctv.db"
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=database_path,
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "runtime" / "cctv.jsonl",
    )
    body = {
        "name": "Restricted entrance",
        "source_name": "sample.mp4",
        "rule_type": "intrusion",
        "class_name": "person",
        "geometry": {
            "type": "polygon",
            "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        },
        "parameters": {"cooldown_seconds": 3},
    }

    with TestClient(create_app(settings)) as client:
        supported = client.get("/rule-types")
        created = client.post("/rules", json=body)
        duplicate = client.post("/rules", json=body)
        rule_id = created.json()["id"]
        listed = client.get(
            "/rules",
            params={"source_name": "SAMPLE.MP4", "rule_type": "INTRUSION", "enabled": True},
        )
        detail = client.get(f"/rules/{rule_id}")
        disabled = client.patch(f"/rules/{rule_id}/enabled", json={"enabled": False})
        mapped = client.put(
            f"/rules/{rule_id}/hiperwall",
            json={
                "content_name": "Entrance camera",
                "zone_id": "Alert Zone",
                "layout": {
                    "mode": "percent",
                    "x": 0,
                    "y": 0,
                    "width": 50,
                    "height": 100,
                },
                "display_seconds": 20,
            },
        )
        invalid_mapping = client.put(
            f"/rules/{rule_id}/hiperwall",
            json={"content_name": "Camera", "content_uuid": "duplicate"},
        )
        cleared_mapping = client.delete(f"/rules/{rule_id}/hiperwall")

        repository = DetectionRepository(database_path)
        run_id = repository.create_analysis_run(
            source_type="local_video",
            source_name="sample.mp4",
            model_name="model.onnx",
            model_sha256="b" * 64,
            device="cpu",
            input_size=640,
            confidence_threshold=0.25,
            nms_threshold=0.45,
            sample_fps=2,
            run_id="run-1",
        ).id
        result = FrameDetections(
            source_index=0,
            sample_index=0,
            timestamp_seconds=0,
            frame_width=100,
            frame_height=100,
            inference_seconds=0.01,
            detections=(Detection(0, "person", 0.9, BoundingBox(40, 30, 60, 60), track_id=1),),
        )
        repository.save_frame(
            run_id,
            result,
            rule_events=(
                RuleEvent(
                    rule_id=rule_id,
                    track_id=1,
                    event_type="intrusion_started",
                    event_state="started",
                    occurred_at_seconds=0,
                    class_name="person",
                    confidence=0.9,
                    payload={"point": [0.5, 0.6]},
                ),
            ),
        )
        events = client.get(
            "/rule-events",
            params={
                "rule_id": rule_id,
                "analysis_run_id": run_id,
                "track_id": 1,
                "event_type": "intrusion_started",
            },
        )

        assert supported.status_code == 200
        assert supported.json()["items"] == [
            "intrusion",
            "line_crossing",
            "loitering",
            "visual_color",
        ]
        assert created.status_code == 201
        assert created.json()["geometry"] == body["geometry"]
        assert duplicate.status_code == 409
        assert listed.status_code == 200
        assert listed.json()["total"] == 1
        assert detail.status_code == 200
        assert detail.json()["name"] == body["name"]
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        assert mapped.status_code == 200
        assert mapped.json()["parameters"]["hiperwall"]["zone_id"] == "Alert Zone"
        assert invalid_mapping.status_code == 422
        assert "hiperwall" not in cleared_mapping.json()["parameters"]
        assert events.status_code == 200
        assert events.json()["total"] == 1
        assert events.json()["items"][0]["payload"]["point"] == [0.5, 0.6]

        assert client.get("/rules/missing").status_code == 404
        assert client.patch("/rules/missing/enabled", json={"enabled": True}).status_code == 404
        assert (
            client.put(
                "/rules/missing/hiperwall",
                json={"content_name": "Camera"},
            ).status_code
            == 404
        )
        assert client.delete("/rules/missing/hiperwall").status_code == 404
        assert client.get("/rule-events", params={"limit": 101}).status_code == 422
        assert (
            client.get(
                "/rule-events",
                params={"occurred_from_seconds": 2, "occurred_to_seconds": 1},
            ).status_code
            == 422
        )


def test_rules_api_rejects_invalid_or_unknown_evaluator_configuration(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=tmp_path / "cctv.db",
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "cctv.jsonl",
    )
    base = {
        "name": "Invalid rule",
        "source_name": "camera",
        "class_name": "person",
        "parameters": {},
    }
    with TestClient(create_app(settings)) as client:
        invalid_polygon = client.post(
            "/rules",
            json={
                **base,
                "rule_type": "intrusion",
                "geometry": {"type": "polygon", "points": [[0, 0], [1, 1]]},
            },
        )
        unknown = client.post(
            "/rules",
            json={**base, "rule_type": "future_rule", "geometry": {}},
        )

    assert invalid_polygon.status_code == 422
    assert unknown.status_code == 422
    assert "unsupported rule_type" in unknown.json()["detail"]


def test_rules_api_creates_and_validates_visual_color_rules(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=tmp_path / "cctv.db",
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "cctv.jsonl",
    )
    body = {
        "name": "Red shirt",
        "source_name": "camera",
        "rule_type": "visual_color",
        "class_name": "person",
        "geometry": {},
        "parameters": {
            "target_color": "red",
            "window_size": 5,
            "minimum_matches": 3,
            "minimum_color_confidence": 0.35,
            "cooldown_seconds": 30,
        },
    }
    with TestClient(create_app(settings)) as client:
        created = client.post("/rules", json=body)
        mapped = client.put(
            f"/rules/{created.json()['id']}/hiperwall",
            json={
                "content_name": "Lobby camera",
                "zone_id": "Alert Zone",
                "display_seconds": 30,
            },
        )
        invalid_votes = client.post(
            "/rules",
            json={
                **body,
                "name": "Invalid votes",
                "parameters": {**body["parameters"], "minimum_matches": 6},
            },
        )
        invalid_color = client.post(
            "/rules",
            json={
                **body,
                "name": "Invalid color",
                "parameters": {**body["parameters"], "target_color": "transparent"},
            },
        )

    assert created.status_code == 201
    assert created.json()["parameters"]["target_color"] == "red"
    assert mapped.status_code == 200
    assert mapped.json()["parameters"]["hiperwall"]["zone_id"] == "Alert Zone"
    assert invalid_votes.status_code == 422
    assert "minimum_matches" in invalid_votes.json()["detail"]
    assert invalid_color.status_code == 422
    assert "target_color" in invalid_color.json()["detail"]
