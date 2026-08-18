import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import DetectionRepository, RuleRepository, connect_database, initialize_database
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.rules import RuleEvent


def _tracked_frame() -> FrameDetections:
    return FrameDetections(
        source_index=10,
        sample_index=1,
        timestamp_seconds=0.5,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.02,
        detections=(
            Detection(
                class_id=0,
                label="person",
                confidence=0.91,
                box=BoundingBox(10, 20, 110, 220),
                track_id=1,
            ),
        ),
    )


def _create_run(repository: DetectionRepository) -> str:
    return repository.create_analysis_run(
        source_type="local_video",
        source_name="sample.mp4",
        model_name="model.onnx",
        model_sha256="a" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
        run_id="run-1",
    ).id


def test_rule_repository_stores_generic_configuration_and_filters(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = RuleRepository(database_path)

    rule = repository.create_rule(
        rule_id="rule-1",
        name="Restricted area",
        source_name="sample.mp4",
        rule_type="intrusion",
        class_name="person",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
        parameters={"cooldown_seconds": 5},
    )
    repository.create_rule(
        rule_id="rule-2",
        name="Exit line",
        source_name="camera",
        rule_type="line_crossing",
        geometry={"type": "line", "points": [[0, 0], [1, 1]]},
        enabled=False,
    )

    assert rule.geometry["type"] == "polygon"
    assert repository.get_rule("rule-1") == rule
    assert repository.list_rules(source_name="SAMPLE.MP4", enabled=True).total == 1
    assert repository.list_rules(rule_type="LINE_CROSSING", enabled=False).total == 1
    assert [item.id for item in repository.list_rule_definitions(source_name="sample.mp4")] == [
        "rule-1"
    ]
    assert repository.set_rule_enabled("rule-1", False).enabled is False
    assert repository.list_rule_definitions(source_name="sample.mp4") == ()


def test_rule_events_are_saved_atomically_with_the_analyzed_frame(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    rules = RuleRepository(database_path)
    rule = rules.create_rule(
        rule_id="rule-1",
        name="Restricted area",
        source_name="sample.mp4",
        rule_type="intrusion",
        geometry={"type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
    )
    detections = DetectionRepository(database_path)
    run_id = _create_run(detections)
    event = RuleEvent(
        id="event-1",
        rule_id=rule.id,
        track_id=1,
        event_type="intrusion_started",
        event_state="started",
        occurred_at_seconds=0.5,
        class_name="person",
        confidence=0.91,
        payload={"point": [0.1, 0.6]},
    )

    frame_id = detections.save_frame(run_id, _tracked_frame(), rule_events=(event,))
    page = rules.list_events(
        rule_id=rule.id,
        analysis_run_id=run_id,
        track_id=1,
        event_type="intrusion_started",
        event_state="started",
        occurred_from_seconds=0.4,
        occurred_to_seconds=0.6,
    )

    assert page.total == 1
    assert page.items[0].id == event.id
    assert page.items[0].analyzed_frame_id == frame_id
    assert page.items[0].rule_name == "Restricted area"
    assert page.items[0].payload == {"point": [0.1, 0.6]}

    invalid_event = RuleEvent(
        rule_id="missing-rule",
        track_id=2,
        event_type="intrusion_started",
        event_state="started",
        occurred_at_seconds=1,
        class_name="person",
        confidence=0.8,
    )
    invalid_frame = FrameDetections(
        source_index=20,
        sample_index=2,
        timestamp_seconds=1,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.02,
        detections=(Detection(0, "person", 0.8, BoundingBox(20, 20, 120, 220), track_id=2),),
    )
    with pytest.raises(sqlite3.IntegrityError):
        detections.save_frame(run_id, invalid_frame, rule_events=(invalid_event,))

    with closing(connect_database(database_path)) as connection:
        frames = connection.execute("SELECT COUNT(*) FROM analyzed_frames").fetchone()[0]
        stored_detections = connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        tracks = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    assert (frames, stored_detections, tracks) == (1, 1, 1)
