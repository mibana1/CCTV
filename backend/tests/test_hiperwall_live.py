from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from cctv.core.settings import Settings
from cctv.db import (
    DetectionRepository,
    DisplayActionRepository,
    RuleRepository,
    initialize_database,
)
from cctv.hiperwall import (
    HiperwallActionWorker,
    HiperwallClient,
    HiperwallLivePlanner,
    HiperwallMappingError,
    HiperwallRequestError,
    validate_rule_hiperwall_mapping,
)
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.rules import RuleDefinition, RuleEvent

CONTENT_XML = """\
<Objects>
  <Object type="Streamer">
    <name>Entrance &amp; Lobby</name>
    <type>Streamer</type>
    <uuid>source-123</uuid>
  </Object>
</Objects>
"""

ZONES_XML = """\
<Zones>
  <Zone>
    <name>Alert Zone</name>
    <id>Alert Zone</id>
  </Zone>
</Zones>
"""


def _inventory_handler(
    requests: list[httpx.Request],
    *,
    content_xml: str = CONTENT_XML,
    zones_xml: str = ZONES_XML,
    command_status: int = 200,
):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/hello":
            return httpx.Response(200, text="Hiperwall,2025R1,Token,Default")
        body = request.content.decode("utf-8")
        if '<action type="list"' in body:
            return httpx.Response(200, text=content_xml)
        if '<action type="walls"' in body:
            return httpx.Response(200, text=zones_xml)
        return httpx.Response(command_status, text="<Response />")

    return handler


def _settings(database_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        app_mode="live",
        hiperwall_base_url="http://hiperwall-host:8000",
        hiperwall_auth_mode="none",
        database_path=database_path,
        **overrides,
    )


def _rule(rule_id: str = "rule-1") -> RuleDefinition:
    return RuleDefinition(
        id=rule_id,
        name="Restricted entrance",
        source_name="camera-1",
        rule_type="intrusion",
        class_name="person",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
        parameters={
            "hiperwall": {
                "content_name": "Entrance & Lobby",
                "zone_id": "Alert Zone",
                "layout": {
                    "mode": "pixels",
                    "x": -100,
                    "y": 20,
                    "width": 1280,
                    "height": 720,
                },
                "display_seconds": 15,
            }
        },
    )


def _event(*, state: str, event_id: str, track_id: int = 7) -> RuleEvent:
    return RuleEvent(
        id=event_id,
        rule_id="rule-1",
        track_id=track_id,
        event_type=f"intrusion_{state}",
        event_state=state,
        occurred_at_seconds=1.5,
        class_name="person",
        confidence=0.93,
        payload={"point": [0.5, 0.8]},
    )


def _frame() -> FrameDetections:
    return FrameDetections(
        source_index=0,
        sample_index=0,
        timestamp_seconds=1.5,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.01,
        detections=(Detection(0, "person", 0.93, BoundingBox(100, 50, 220, 300), track_id=7),),
    )


def test_live_planner_maps_lifecycle_and_schedules_point_event_close(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "cctv.db", hiperwall_executor_enabled=False)
    planner = HiperwallLivePlanner(settings, (_rule(),))

    started, ended, occurred = (
        _event(state="started", event_id="started"),
        _event(state="ended", event_id="ended"),
        _event(state="occurred", event_id="occurred"),
    )
    actions = planner.plan(
        (started, ended, occurred),
        analysis_run_id="run-1",
        source_name="camera-1",
    )

    assert [action.action_type for action in actions] == [
        "open_source",
        "restore_layout",
        "open_source",
    ]
    assert actions[0].request["instance_id"] == actions[1].request["instance_id"]
    assert actions[2].request["close_after_seconds"] == 15
    assert actions[0].request["target"] == {
        "selector": "name",
        "value": "Entrance & Lobby",
        "zone_id": "Alert Zone",
        "layout": {"mode": "pixels", "x": -100.0, "y": 20.0, "width": 1280.0, "height": 720.0},
    }
    assert all(action.mode == "live" and action.status == "pending" for action in actions)


def test_mapping_requires_one_content_and_bounded_percent_layout() -> None:
    missing_content = _rule()
    missing_content.parameters["hiperwall"].pop("content_name")
    with pytest.raises(HiperwallMappingError, match="exactly one"):
        validate_rule_hiperwall_mapping(missing_content)

    invalid_layout = _rule()
    invalid_layout.parameters["hiperwall"]["layout"] = {
        "mode": "percent",
        "x": 80,
        "y": 0,
        "width": 30,
        "height": 100,
    }
    with pytest.raises(HiperwallMappingError, match="remain inside"):
        validate_rule_hiperwall_mapping(invalid_layout)


def test_client_sends_token_authenticated_open_and_close_xml() -> None:
    requests: list[httpx.Request] = []
    http_client = httpx.Client(transport=httpx.MockTransport(_inventory_handler(requests)))
    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        auth_mode="token",
        user="cctv_bridge",
        token="secret-token",
        http_client=http_client,
    )
    open_request = {
        "operation": "open_source",
        "instance_id": "cctv-camera-1-abc",
        "target": {
            "selector": "name",
            "value": "Entrance & Lobby",
            "zone_id": "Alert Zone",
            "layout": {"mode": "percent", "x": 0, "y": 0, "width": 50, "height": 100},
        },
        "label": "Intrusion <person>",
    }

    opened = client.execute(open_request)
    closed = client.execute({"operation": "restore_layout", "instance_id": "cctv-camera-1-abc"})

    xml_requests = [request for request in requests if request.method == "POST"]
    open_request_sent = next(
        request
        for request in xml_requests
        if '<command type="open"' in request.content.decode("utf-8")
    )
    close_request_sent = next(
        request
        for request in xml_requests
        if '<command type="close"' in request.content.decode("utf-8")
    )
    open_xml = open_request_sent.content.decode("utf-8")
    close_xml = close_request_sent.content.decode("utf-8")
    assert open_request_sent.url == "http://hiperwall-host:8000/xmlcommand"
    assert open_request_sent.headers["content-type"] == "application/xml; charset=utf-8"
    assert '<auth type="token">' in open_xml
    assert "<user>cctv_bridge</user>" in open_xml
    assert "<token>secret-token</token>" in open_xml
    assert "<name>Entrance &amp; Lobby</name>" in open_xml
    assert "<position>Alert Zone:0,0,0.5,1</position>" in open_xml
    assert "<label>Intrusion &lt;person&gt;</label>" in open_xml
    assert '<command type="close">' in close_xml
    assert opened["external_request_sent"] is True
    assert closed["operation"] == "restore_layout"
    assert len(requests) == 5


def test_client_rejects_removed_content_before_open_command() -> None:
    requests: list[httpx.Request] = []
    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        http_client=httpx.Client(transport=httpx.MockTransport(_inventory_handler(requests))),
    )

    with pytest.raises(HiperwallRequestError) as captured:
        client.execute(
            {
                "operation": "open_source",
                "instance_id": "cctv-camera-1-missing-content",
                "target": {"selector": "uuid", "value": "removed-source"},
            }
        )

    assert captured.value.code == "hiperwall_content_not_found"
    assert captured.value.retryable is False
    assert not any(
        '<command type="open"' in request.content.decode("utf-8")
        for request in requests
        if request.method == "POST"
    )


def test_client_rejects_removed_zone_before_open_command() -> None:
    requests: list[httpx.Request] = []
    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        http_client=httpx.Client(transport=httpx.MockTransport(_inventory_handler(requests))),
    )

    with pytest.raises(HiperwallRequestError) as captured:
        client.execute(
            {
                "operation": "open_source",
                "instance_id": "cctv-camera-1-missing-zone",
                "target": {
                    "selector": "name",
                    "value": "Entrance & Lobby",
                    "zone_id": "Removed Zone",
                },
            }
        )

    assert captured.value.code == "hiperwall_zone_not_found"
    assert captured.value.retryable is False
    assert not any(
        '<command type="open"' in request.content.decode("utf-8")
        for request in requests
        if request.method == "POST"
    )


def test_client_distinguishes_inventory_authentication_failure() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/hello":
            return httpx.Response(200, text="Hiperwall,2025R1,Token,Default")
        return httpx.Response(403, text="Forbidden")

    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        auth_mode="token",
        user="cctv_bridge",
        token="wrong-token",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(HiperwallRequestError) as captured:
        client.execute(
            {
                "operation": "open_source",
                "instance_id": "cctv-camera-1-auth",
                "target": {"selector": "name", "value": "Entrance & Lobby"},
            }
        )

    assert captured.value.code == "hiperwall_auth_failed"
    assert captured.value.retryable is False


def test_client_distinguishes_forbidden_command_after_valid_preflight() -> None:
    requests: list[httpx.Request] = []
    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_inventory_handler(requests, command_status=403))
        ),
    )

    with pytest.raises(HiperwallRequestError) as captured:
        client.execute(
            {
                "operation": "open_source",
                "instance_id": "cctv-camera-1-forbidden",
                "target": {
                    "selector": "name",
                    "value": "Entrance & Lobby",
                    "zone_id": "Alert Zone",
                },
            }
        )

    assert captured.value.code == "hiperwall_command_forbidden"
    assert captured.value.retryable is False


def test_client_rejects_http_200_business_error() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, text="<Response><Error>bad target</Error></Response>"
            )
        )
    )
    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        http_client=http_client,
    )

    with pytest.raises(HiperwallRequestError) as captured:
        client.execute(
            {
                "operation": "restore_layout",
                "instance_id": "cctv-camera-1-abc",
            }
        )

    assert captured.value.code == "hiperwall_business_error"
    assert captured.value.retryable is False


def test_durable_worker_executes_persisted_live_action(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "cctv.db"
    initialize_database(database_path)
    rules = RuleRepository(database_path)
    rule = rules.create_rule(
        rule_id="rule-1",
        name="Restricted entrance",
        source_name="camera-1",
        rule_type="intrusion",
        class_name="person",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
        parameters=_rule().parameters,
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
    settings = _settings(database_path)
    event = _event(state="occurred", event_id="occurred")
    actions = HiperwallLivePlanner(settings, (rule.to_definition(),)).plan(
        (event,),
        analysis_run_id=run.id,
        source_name="camera-1",
    )
    detections.save_frame(run.id, _frame(), rule_events=(event,), display_actions=actions)

    sent: list[httpx.Request] = []

    client = HiperwallClient(
        base_url=settings.hiperwall_base_url or "",
        http_client=httpx.Client(transport=httpx.MockTransport(_inventory_handler(sent))),
    )
    worker = HiperwallActionWorker(settings, client=client)

    assert worker.process_once() is True
    assert worker.process_once() is False

    record = DisplayActionRepository(database_path).get(actions[0].id)
    assert record is not None
    assert record.status == "succeeded"
    assert record.attempt_count == 1
    assert record.last_attempt_at is not None
    assert record.completed_at is not None
    assert record.result["external_request_sent"] is True
    assert len(sent) == 4
    assert (
        sum(
            '<command type="open"' in request.content.decode("utf-8")
            for request in sent
            if request.method == "POST"
        )
        == 1
    )
    assert record.updated_at <= datetime.now(UTC)
    scheduled = DisplayActionRepository(database_path).list(
        rule_event_id=event.id,
        status="pending",
    )
    assert scheduled.total == 1
    assert scheduled.items[0].action_type == "restore_layout"
    assert scheduled.items[0].available_at > record.completed_at
