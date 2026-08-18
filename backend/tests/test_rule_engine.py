from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.rules import RuleDefinition, RuleEngine, RuleEvent, UnsupportedRuleTypeError


def rule(
    rule_type: str,
    geometry: dict[str, object],
    *,
    parameters: dict[str, object] | None = None,
    class_name: str | None = "person",
) -> RuleDefinition:
    return RuleDefinition(
        id=f"{rule_type}-rule",
        name=f"Test {rule_type}",
        source_name="camera-1",
        rule_type=rule_type,
        class_name=class_name,
        geometry=geometry,
        parameters=parameters or {},
    )


def frame(
    sample_index: int,
    timestamp_seconds: float,
    *detections: Detection,
) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index,
        sample_index=sample_index,
        timestamp_seconds=timestamp_seconds,
        frame_width=100,
        frame_height=100,
        inference_seconds=0.01,
        detections=detections,
    )


def tracked_detection(
    track_id: int,
    *,
    foot_x: int,
    foot_y: int,
    class_name: str = "person",
) -> Detection:
    return Detection(
        class_id=0,
        label=class_name,
        confidence=0.9,
        box=BoundingBox(x1=foot_x - 5, y1=foot_y - 20, x2=foot_x + 5, y2=foot_y),
        track_id=track_id,
    )


def test_intrusion_emits_one_started_and_ended_lifecycle() -> None:
    engine = RuleEngine(
        [
            rule(
                "intrusion",
                {
                    "type": "polygon",
                    "points": [[0.4, 0.4], [0.8, 0.4], [0.8, 0.8], [0.4, 0.8]],
                },
            )
        ]
    )

    assert engine.process(frame(0, 0, tracked_detection(1, foot_x=20, foot_y=60))) == ()
    started = engine.process(frame(1, 0.5, tracked_detection(1, foot_x=50, foot_y=60)))
    assert engine.process(frame(2, 1.0, tracked_detection(1, foot_x=55, foot_y=60))) == ()
    ended = engine.process(frame(3, 1.5, tracked_detection(1, foot_x=90, foot_y=60)))

    assert [event.event_type for event in started] == ["intrusion_started"]
    assert started[0].event_state == "started"
    assert started[0].track_id == 1
    assert started[0].payload["point"] == [0.5, 0.6]
    assert [event.event_type for event in ended] == ["intrusion_ended"]
    assert ended[0].payload["reason"] == "outside"


def test_line_crossing_honors_direction_and_does_not_repeat_near_line() -> None:
    engine = RuleEngine(
        [
            rule(
                "line_crossing",
                {"type": "line", "points": [[0.5, 0.2], [0.5, 0.9]]},
                parameters={"direction": "positive_to_negative", "cooldown_seconds": 2},
            )
        ]
    )

    assert engine.process(frame(0, 0, tracked_detection(1, foot_x=30, foot_y=60))) == ()
    assert engine.process(frame(1, 0.5, tracked_detection(1, foot_x=50, foot_y=60))) == ()
    crossed = engine.process(frame(2, 1.0, tracked_detection(1, foot_x=70, foot_y=60)))
    assert engine.process(frame(3, 1.5, tracked_detection(1, foot_x=30, foot_y=60))) == ()

    assert len(crossed) == 1
    assert crossed[0].event_type == "line_crossed"
    assert crossed[0].event_state == "occurred"
    assert crossed[0].payload["direction"] == "positive_to_negative"


def test_loitering_uses_source_time_and_ends_after_missing_tolerance() -> None:
    engine = RuleEngine(
        [
            rule(
                "loitering",
                {
                    "type": "polygon",
                    "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                },
                parameters={"duration_seconds": 2, "missing_tolerance_seconds": 2},
            )
        ]
    )

    assert engine.process(frame(0, 0, tracked_detection(1, foot_x=50, foot_y=50))) == ()
    assert engine.process(frame(1, 1, tracked_detection(1, foot_x=51, foot_y=50))) == ()
    started = engine.process(frame(2, 2, tracked_detection(1, foot_x=52, foot_y=50)))
    assert engine.process(frame(3, 3)) == ()
    ended = engine.process(frame(4, 5))

    assert [event.event_type for event in started] == ["loitering_started"]
    assert started[0].payload["duration_seconds"] == 2
    assert [event.event_type for event in ended] == ["loitering_ended"]
    assert ended[0].payload["reason"] == "missing"


def test_engine_filters_target_class_and_skips_untracked_detections() -> None:
    engine = RuleEngine(
        [
            rule(
                "intrusion",
                {
                    "type": "polygon",
                    "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
                class_name="person",
            )
        ]
    )
    car = tracked_detection(1, foot_x=50, foot_y=50, class_name="car")
    untracked_person = Detection(
        class_id=0,
        label="person",
        confidence=0.9,
        box=BoundingBox(45, 30, 55, 50),
    )

    assert engine.process(frame(0, 0, car, untracked_person)) == ()


def test_engine_accepts_custom_evaluator_without_dispatch_changes() -> None:
    class CustomEvaluator:
        rule_type = "custom"

        def validate(self, definition: RuleDefinition) -> None:
            del definition

        def evaluate(self, definition, timestamp_seconds, samples):
            return tuple(
                RuleEvent(
                    rule_id=definition.id,
                    track_id=sample.track_id,
                    event_type="custom_detected",
                    event_state="occurred",
                    occurred_at_seconds=timestamp_seconds,
                    class_name=sample.class_name,
                    confidence=sample.confidence,
                )
                for sample in samples
            )

    engine = RuleEngine(
        [rule("custom", {}, class_name=None)],
        evaluators=[CustomEvaluator()],
    )

    events = engine.process(frame(0, 0, tracked_detection(3, foot_x=50, foot_y=50)))

    assert engine.supported_rule_types == ("custom",)
    assert [event.event_type for event in events] == ["custom_detected"]


def test_engine_rejects_unsupported_rule_type() -> None:
    try:
        RuleEngine([rule("unknown", {})])
    except UnsupportedRuleTypeError as error:
        assert "unsupported rule_type" in str(error)
    else:
        raise AssertionError("unsupported rule type was accepted")
