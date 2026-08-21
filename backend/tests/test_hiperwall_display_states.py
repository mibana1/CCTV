from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from cctv.core.settings import Settings
from cctv.db import (
    DetectionRepository,
    DisplayActionRepository,
    DisplayStateRepository,
    RuleRepository,
    initialize_database,
)
from cctv.hiperwall import HiperwallLivePlanner
from cctv.hiperwall.live import _scheduled_close
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.rules import RuleEvent


def _settings(database_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_mode="live",
        hiperwall_base_url="http://hiperwall-host:8000",
        hiperwall_auth_mode="none",
        database_path=database_path,
    )


def _seed(database_path: Path):
    initialize_database(database_path)
    rule = RuleRepository(database_path).create_rule(
        rule_id="rule-1",
        name="Gray shirt",
        source_name="camera-1",
        rule_type="visual_color",
        class_name="person",
        geometry={},
        parameters={
            "target_color": "gray",
            "window_size": 5,
            "minimum_matches": 3,
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
    planner = HiperwallLivePlanner(_settings(database_path), (rule.to_definition(),))
    return rule, detections, run, planner


def _event(event_id: str, track_id: int, timestamp: float) -> RuleEvent:
    return RuleEvent(
        id=event_id,
        rule_id="rule-1",
        track_id=track_id,
        event_type="upper_body_color_detected",
        event_state="occurred",
        occurred_at_seconds=timestamp,
        class_name="person",
        confidence=0.9,
        payload={"target_color": "gray"},
    )


def _frame(sample_index: int, track_id: int) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index,
        sample_index=sample_index,
        timestamp_seconds=float(sample_index),
        frame_width=640,
        frame_height=360,
        inference_seconds=0.01,
        detections=(
            Detection(
                0,
                "person",
                0.9,
                BoundingBox(100, 50, 220, 300),
                track_id=track_id,
            ),
        ),
    )


def _save_candidate(detections, run, planner, *, index: int, track_id: int):
    event = _event(f"event-{index}", track_id, float(index))
    actions = planner.plan(
        (event,),
        analysis_run_id=run.id,
        source_name="camera-1",
    )
    outcome = detections.save_frame_with_outcome(
        run.id,
        _frame(index, track_id),
        active_track_ids={track_id},
        rule_events=(event,),
        display_actions=actions,
    )
    return event, actions[0], outcome


def test_rule_camera_gate_waits_for_close_and_cooldown_before_rearming(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    rule, detections, run, planner = _seed(database_path)
    actions = DisplayActionRepository(database_path)
    states = DisplayStateRepository(database_path)

    first_event, first_open, first = _save_candidate(
        detections,
        run,
        planner,
        index=0,
        track_id=1,
    )
    _, second_open, second = _save_candidate(
        detections,
        run,
        planner,
        index=1,
        track_id=99,
    )

    assert first.persisted_rule_event_ids == (first_event.id,)
    assert first.persisted_display_action_ids == (first_open.id,)
    assert second.suppressed_rule_event_ids == ("event-1",)
    assert second.suppressed_display_action_ids == (second_open.id,)
    state = states.get(rule.id, "camera-1")
    assert state is not None
    assert state.state == "DISPLAYING"
    assert state.active_rule_event_id == first_event.id

    claimed_open = actions.claim_next()
    assert claimed_open is not None
    scheduled_close = _scheduled_close(claimed_open)
    assert scheduled_close is not None
    actions.mark_succeeded(
        claimed_open.id,
        result={"external_request_sent": True, "operation": "open_source"},
        followup_action=scheduled_close,
    )
    displaying = states.get(rule.id, "camera-1")
    assert displaying is not None
    assert displaying.state == "DISPLAYING"
    assert displaying.opened_at is not None
    assert displaying.close_action_id == scheduled_close.id

    claimed_close = actions.claim_next(now=scheduled_close.available_at + timedelta(seconds=1))
    assert claimed_close is not None
    assert claimed_close.id == scheduled_close.id
    actions.mark_succeeded(
        claimed_close.id,
        result={"external_request_sent": True, "operation": "restore_layout"},
    )
    cooldown = states.get(rule.id, "camera-1")
    assert cooldown is not None
    assert cooldown.state == "COOLDOWN"
    assert cooldown.close_succeeded_at is not None
    assert cooldown.cooldown_until is not None
    assert (cooldown.cooldown_until - cooldown.close_succeeded_at).total_seconds() == 7

    _, cooldown_open, during_cooldown = _save_candidate(
        detections,
        run,
        planner,
        index=2,
        track_id=100,
    )
    assert during_cooldown.suppressed_rule_event_ids == ("event-2",)
    assert during_cooldown.suppressed_display_action_ids == (cooldown_open.id,)

    states.release_expired_cooldowns(now=cooldown.cooldown_until + timedelta(seconds=1))
    idle = states.get(rule.id, "camera-1")
    assert idle is not None
    assert idle.state == "IDLE"
    assert idle.active_rule_event_id is None

    third_event, third_open, rearmed = _save_candidate(
        detections,
        run,
        planner,
        index=3,
        track_id=101,
    )
    assert rearmed.persisted_rule_event_ids == (third_event.id,)
    assert rearmed.persisted_display_action_ids == (third_open.id,)
    assert RuleRepository(database_path).list_events(rule_id=rule.id).total == 2


def test_terminal_close_failure_keeps_gate_displaying(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    rule, detections, run, planner = _seed(database_path)
    actions = DisplayActionRepository(database_path)
    states = DisplayStateRepository(database_path)

    _save_candidate(detections, run, planner, index=0, track_id=1)
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

    state = states.get(rule.id, "camera-1")
    assert state is not None
    assert state.state == "DISPLAYING"
    assert state.close_action_id == claimed_close.id


def test_preflight_open_failure_returns_gate_to_idle(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    rule, detections, run, planner = _seed(database_path)
    actions = DisplayActionRepository(database_path)
    states = DisplayStateRepository(database_path)

    _save_candidate(detections, run, planner, index=0, track_id=1)
    claimed_open = actions.claim_next()
    assert claimed_open is not None
    actions.mark_failed(
        claimed_open.id,
        error_code="hiperwall_content_not_found",
        error_message="missing",
        retry_after_seconds=None,
    )

    state = states.get(rule.id, "camera-1")
    assert state is not None
    assert state.state == "IDLE"
    assert state.active_rule_event_id is None


def test_concurrent_new_tracks_admit_only_one_event(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    rule, detections, first_run, planner = _seed(database_path)
    second_run = detections.create_analysis_run(
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
        run_id="run-2",
    )

    def persist(run, event_id: str, track_id: int):
        event = _event(event_id, track_id, 0)
        actions = planner.plan(
            (event,),
            analysis_run_id=run.id,
            source_name="camera-1",
        )
        return detections.save_frame_with_outcome(
            run.id,
            _frame(0, track_id),
            active_track_ids={track_id},
            rule_events=(event,),
            display_actions=actions,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(persist, first_run, "concurrent-1", 1),
            executor.submit(persist, second_run, "concurrent-2", 2),
        )
        outcomes = tuple(future.result() for future in futures)

    assert sum(len(item.persisted_rule_event_ids) for item in outcomes) == 1
    assert sum(len(item.suppressed_rule_event_ids) for item in outcomes) == 1
    assert RuleRepository(database_path).list_events(rule_id=rule.id).total == 1
    assert DisplayActionRepository(database_path).list().total == 1
