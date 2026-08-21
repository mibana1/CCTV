from datetime import UTC, datetime, timedelta
from pathlib import Path

from cctv.core.settings import Settings
from cctv.db import (
    DetectionRepository,
    DisplayActionRepository,
    DisplayStateRepository,
    RuleRepository,
    initialize_database,
)
from cctv.hiperwall import HiperwallLivePlanner, HiperwallReconciler, HiperwallRequestError
from cctv.hiperwall.live import _scheduled_close
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.rules import RuleEvent


class _FakeHiperwallClient:
    def __init__(self, instances: set[str], *, fail_close: bool = False) -> None:
        self.instances = set(instances)
        self.fail_close = fail_close
        self.executed: list[dict[str, object]] = []

    def inventory(self) -> dict[str, object]:
        return {
            "contents": [
                {
                    "name": "Alert",
                    "instances": [{"id": instance_id} for instance_id in self.instances],
                }
            ]
        }

    def execute(self, request: dict[str, object]) -> dict[str, object]:
        self.executed.append(dict(request))
        if self.fail_close:
            raise HiperwallRequestError("hiperwall_timeout", "close timed out")
        instance_id = str(request["instance_id"])
        self.instances.discard(instance_id)
        return {
            "external_request_sent": True,
            "operation": request["operation"],
            "instance_id": instance_id,
        }


def _settings(database_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_mode="live",
        hiperwall_base_url="http://hiperwall-host:8000",
        hiperwall_auth_mode="none",
        hiperwall_reconciliation_grace_seconds=0,
        database_path=database_path,
    )


def _seed(database_path: Path):
    initialize_database(database_path)
    settings = _settings(database_path)
    rule = RuleRepository(database_path).create_rule(
        rule_id="rule-1",
        name="Restricted entrance",
        source_name="camera-1",
        rule_type="intrusion",
        class_name="person",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
        parameters={
            "cooldown_seconds": 7,
            "hiperwall": {
                "content_name": "Alert",
                "zone_id": "Left",
                "display_seconds": 15,
            },
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
    event = RuleEvent(
        id="event-1",
        rule_id=rule.id,
        track_id=7,
        event_type="intrusion_occurred",
        event_state="occurred",
        occurred_at_seconds=1,
        class_name="person",
        confidence=0.9,
        payload={},
    )
    frame = FrameDetections(
        source_index=0,
        sample_index=0,
        timestamp_seconds=1,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.01,
        detections=(
            Detection(
                0,
                "person",
                0.9,
                BoundingBox(100, 50, 220, 300),
                track_id=7,
            ),
        ),
    )
    planned = HiperwallLivePlanner(settings, (rule.to_definition(),)).plan(
        (event,),
        analysis_run_id=run.id,
        source_name="camera-1",
    )
    detections.save_frame(
        run.id,
        frame,
        active_track_ids={7},
        rule_events=(event,),
        display_actions=planned,
    )
    return settings, rule, planned[0]


def _terminal_close(database_path: Path):
    settings, rule, planned_open = _seed(database_path)
    actions = DisplayActionRepository(database_path)
    claimed_open = actions.claim_next()
    assert claimed_open is not None
    scheduled_close = _scheduled_close(claimed_open)
    assert scheduled_close is not None
    actions.mark_succeeded(
        claimed_open.id,
        result={"external_request_sent": True},
        followup_action=scheduled_close,
    )
    claimed_close = actions.claim_next(now=scheduled_close.available_at + timedelta(seconds=1))
    assert claimed_close is not None
    actions.mark_failed(
        claimed_close.id,
        error_code="hiperwall_timeout",
        error_message="timeout",
        retry_after_seconds=None,
    )
    return settings, rule, planned_open, claimed_close


def test_reconciliation_adopts_open_observed_after_process_interruption(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    settings, rule, planned_open = _seed(database_path)
    actions = DisplayActionRepository(database_path)
    claimed_open = actions.claim_next()
    assert claimed_open is not None
    assert actions.recover_interrupted(hold_for_reconciliation=True) == 1
    assert actions.claim_next(allow_uncertain_outcomes=False) is None

    instance_id = str(planned_open.request["instance_id"])
    client = _FakeHiperwallClient({instance_id})
    result = HiperwallReconciler(settings, client=client).run_once()

    adopted = actions.get(planned_open.id)
    state = DisplayStateRepository(database_path).get(rule.id, "camera-1")
    scheduled = actions.list(rule_event_id=planned_open.rule_event_id, action_type="restore_layout")
    assert result.adopted_open_count == 1
    assert result.forced_close_attempt_count == 0
    assert adopted is not None and adopted.status == "succeeded"
    assert adopted.result["reconciliation_reason"] == "external_open_observed"
    assert scheduled.total == 1
    assert state is not None and state.close_action_id == scheduled.items[0].id
    assert client.executed == []


def test_reconciliation_force_closes_terminal_close_failure(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    settings, rule, planned_open, failed_close = _terminal_close(database_path)
    instance_id = str(planned_open.request["instance_id"])
    client = _FakeHiperwallClient({instance_id})
    reconciled_at = (failed_close.last_attempt_at or datetime.now(UTC)) + timedelta(seconds=1)

    result = HiperwallReconciler(
        settings,
        client=client,
        now_factory=lambda: reconciled_at,
    ).run_once()

    close_record = DisplayActionRepository(database_path).get(failed_close.id)
    state = DisplayStateRepository(database_path).get(rule.id, "camera-1")
    assert result.forced_close_attempt_count == 1
    assert result.forced_close_succeeded_count == 1
    assert close_record is not None and close_record.status == "succeeded"
    assert close_record.result["reconciliation_reason"] == "forced_close_succeeded"
    assert state is not None and state.state == "COOLDOWN"
    assert client.executed == [
        {"operation": "restore_layout", "instance_id": instance_id}
    ]


def test_reconciliation_keeps_failed_close_for_next_force_attempt(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    settings, rule, planned_open, failed_close = _terminal_close(database_path)
    instance_id = str(planned_open.request["instance_id"])
    client = _FakeHiperwallClient({instance_id}, fail_close=True)
    reconciled_at = (failed_close.last_attempt_at or datetime.now(UTC)) + timedelta(seconds=1)

    result = HiperwallReconciler(
        settings,
        client=client,
        now_factory=lambda: reconciled_at,
    ).run_once()

    close_record = DisplayActionRepository(database_path).get(failed_close.id)
    state = DisplayStateRepository(database_path).get(rule.id, "camera-1")
    assert result.forced_close_failed_count == 1
    assert close_record is not None and close_record.status == "failed"
    assert close_record.last_error_code == "hiperwall_timeout"
    assert state is not None and state.state == "DISPLAYING"


def test_reconciliation_closes_only_orphan_instances_owned_by_cctv(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    settings = _settings(database_path)
    client = _FakeHiperwallClient({"cctv-orphan-1", "operator-manual-1"})

    result = HiperwallReconciler(settings, client=client).run_once()

    assert result.inventory_instance_count == 2
    assert result.managed_external_instance_count == 1
    assert result.forced_close_succeeded_count == 1
    assert client.executed == [
        {"operation": "restore_layout", "instance_id": "cctv-orphan-1"}
    ]
    assert client.instances == {"operator-manual-1"}


def test_reconciliation_marks_failed_close_succeeded_when_instance_is_already_absent(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    settings, rule, _, failed_close = _terminal_close(database_path)
    reconciled_at = (failed_close.last_attempt_at or datetime.now(UTC)) + timedelta(seconds=1)
    client = _FakeHiperwallClient(set())

    result = HiperwallReconciler(
        settings,
        client=client,
        now_factory=lambda: reconciled_at,
    ).run_once()

    close_record = DisplayActionRepository(database_path).get(failed_close.id)
    state = DisplayStateRepository(database_path).get(rule.id, "camera-1")
    assert result.reconciled_close_count == 1
    assert result.forced_close_attempt_count == 0
    assert close_record is not None and close_record.status == "succeeded"
    assert close_record.result["reconciliation_reason"] == "external_instance_already_absent"
    assert state is not None and state.state == "COOLDOWN"
    assert client.executed == []
