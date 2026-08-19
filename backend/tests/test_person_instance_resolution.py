from pathlib import Path

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import (
    DetectionRepository,
    FaceMatchEventInput,
    FaceMatchRepository,
    PersonInstanceRepository,
    initialize_database,
)
from cctv.identity import TrackIdentityResolver
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.main import create_app


def _run(database_path: Path) -> str:
    initialize_database(database_path)
    return DetectionRepository(database_path).create_analysis_run(
        source_type="rtsp",
        source_name="camera",
        detector_type="test",
        model_name="model.onnx",
        model_sha256="a" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
    ).id


def _event(
    *,
    track_id: int,
    sample_index: int,
    timestamp_seconds: float,
    matched: bool,
    similarity: float,
    x: int,
    y: int,
    width: int,
    height: int,
) -> FaceMatchEventInput:
    return FaceMatchEventInput(
        source_index=sample_index * 12,
        sample_index=sample_index,
        source_timestamp_seconds=timestamp_seconds,
        face_index=0,
        track_id=track_id,
        face_x=x,
        face_y=y,
        face_width=width,
        face_height=height,
        detection_confidence=0.9,
        match_status="matched" if matched else "unknown",
        rejection_reason=None if matched else "below_threshold",
        identity_id="emp-001" if matched else None,
        external_id="EMP-001" if matched else None,
        display_name="Employee" if matched else None,
        best_candidate_identity_id="emp-001",
        best_candidate_external_id="EMP-001",
        best_similarity=similarity,
        second_best_similarity=None,
        similarity_threshold=0.45,
        minimum_margin=0.05,
    )


def _tracked_frame(
    *,
    track_id: int,
    sample_index: int,
    timestamp_seconds: float,
) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index * 12,
        sample_index=sample_index,
        timestamp_seconds=timestamp_seconds,
        frame_width=1280,
        frame_height=720,
        inference_seconds=0.01,
        detections=(
            Detection(
                class_id=0,
                label="person",
                confidence=0.9,
                box=BoundingBox(620, 300, 780, 680),
                track_id=track_id,
            ),
        ),
    )


def test_resolver_stitches_unknown_and_matched_track_fragments(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    run_id = _run(database_path)
    faces = FaceMatchRepository(database_path)
    people = PersonInstanceRepository(database_path)
    resolver = TrackIdentityResolver(people)

    observations = (
        _event(
            track_id=15,
            sample_index=10,
            timestamp_seconds=6.08,
            matched=False,
            similarity=0.27,
            x=464,
            y=329,
            width=19,
            height=22,
        ),
        _event(
            track_id=21,
            sample_index=13,
            timestamp_seconds=7.55,
            matched=False,
            similarity=0.31,
            x=378,
            y=355,
            width=23,
            height=26,
        ),
        _event(
            track_id=24,
            sample_index=15,
            timestamp_seconds=8.55,
            matched=False,
            similarity=0.36,
            x=282,
            y=400,
            width=27,
            height=35,
        ),
        _event(
            track_id=26,
            sample_index=16,
            timestamp_seconds=9.01,
            matched=True,
            similarity=0.55,
            x=211,
            y=435,
            width=32,
            height=38,
        ),
        _event(
            track_id=27,
            sample_index=17,
            timestamp_seconds=9.55,
            matched=False,
            similarity=0.26,
            x=116,
            y=477,
            width=34,
            height=41,
        ),
    )

    resolved_ids = []
    for observation in observations:
        stored = faces.save_event(run_id, observation)
        resolved = resolver.resolve(stored)
        assert resolved is not None
        resolved_ids.append(resolved.id)

    assert len(set(resolved_ids)) == 1
    instance = people.get_instance(resolved_ids[0])
    assert instance is not None
    assert instance.status == "matched"
    assert instance.identity_id == "emp-001"
    assert instance.external_id == "EMP-001"
    assert instance.track_count == 5
    assert instance.first_sample_index == 10
    assert instance.last_sample_index == 17

    links = people.list_links(instance.id, limit=100)
    assert [link.track_id for link in links.items] == [15, 21, 24, 26, 27]
    assert [link.linked_by for link in links.items] == [
        "unknown_observation",
        "candidate_stitch",
        "candidate_stitch",
        "candidate_stitch",
        "candidate_stitch",
    ]
    persisted_events = faces.list_events(analysis_run_id=run_id, limit=100)
    assert {event.person_instance_id for event in persisted_events.items} == {instance.id}
    assert people.summarize_run(run_id).matched_instances == 1
    assert people.summarize_run(run_id).unknown_instances == 0


def test_resolver_does_not_force_low_similarity_unknown_into_match(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    run_id = _run(database_path)
    faces = FaceMatchRepository(database_path)
    people = PersonInstanceRepository(database_path)
    resolver = TrackIdentityResolver(people)

    unknown = faces.save_event(
        run_id,
        _event(
            track_id=1,
            sample_index=1,
            timestamp_seconds=1,
            matched=False,
            similarity=0.2,
            x=100,
            y=100,
            width=30,
            height=30,
        ),
    )
    matched = faces.save_event(
        run_id,
        _event(
            track_id=2,
            sample_index=2,
            timestamp_seconds=2,
            matched=True,
            similarity=0.8,
            x=800,
            y=100,
            width=30,
            height=30,
        ),
    )

    unknown_instance = resolver.resolve(unknown)
    matched_instance = resolver.resolve(matched)

    assert unknown_instance is not None
    assert matched_instance is not None
    assert unknown_instance.id != matched_instance.id
    summary = people.summarize_run(run_id)
    assert summary.total_instances == 2
    assert summary.matched_instances == 1
    assert summary.unknown_instances == 1


def test_resolver_keeps_stationary_unknown_person_across_real_camera_gaps(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    run_id = _run(database_path)
    detections = DetectionRepository(database_path)
    faces = FaceMatchRepository(database_path)
    people = PersonInstanceRepository(database_path)
    resolver = TrackIdentityResolver(people)
    observations = (
        (2, 108, 55.021, 688, 513, 0.135, 111, 56.521),
        (11, 118, 60.017, 693, 530, 0.133, 137, 69.735),
        (12, 146, 74.018, 692, 521, 0.090, 155, 79.021),
        (13, 175, 89.015, 693, 523, 0.158, 180, 91.513),
        (14, 196, 99.735, 691, 522, 0.172, 213, 108.774),
        (15, 214, 119.096, 691, 519, 0.202, 216, 120.039),
        (16, 217, 128.341, 687, 508, 0.117, 239, 139.019),
    )

    resolved_ids: list[str] = []
    for (
        track_id,
        face_sample,
        face_timestamp,
        face_x,
        face_y,
        similarity,
        last_track_sample,
        last_track_timestamp,
    ) in observations:
        resolved = resolver.resolve(
            faces.save_event(
                run_id,
                _event(
                    track_id=track_id,
                    sample_index=face_sample,
                    timestamp_seconds=face_timestamp,
                    matched=False,
                    similarity=similarity,
                    x=face_x,
                    y=face_y,
                    width=40,
                    height=52,
                ),
            )
        )
        assert resolved is not None
        resolved_ids.append(resolved.id)
        detections.save_frame(
            run_id,
            _tracked_frame(
                track_id=track_id,
                sample_index=last_track_sample,
                timestamp_seconds=last_track_timestamp,
            ),
        )

    assert len(set(resolved_ids)) == 1
    instance = people.get_instance(resolved_ids[0])
    assert instance is not None
    assert instance.status == "unknown"
    assert instance.track_count == 7
    assert [
        link.track_id for link in people.list_links(instance.id, limit=100).items
    ] == [2, 11, 12, 13, 14, 15, 16]


def test_resolver_does_not_merge_distant_weak_candidates_across_long_gap(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    run_id = _run(database_path)
    detections = DetectionRepository(database_path)
    faces = FaceMatchRepository(database_path)
    people = PersonInstanceRepository(database_path)
    resolver = TrackIdentityResolver(people)

    first = resolver.resolve(
        faces.save_event(
            run_id,
            _event(
                track_id=1,
                sample_index=1,
                timestamp_seconds=1,
                matched=False,
                similarity=0.1,
                x=100,
                y=100,
                width=30,
                height=30,
            ),
        )
    )
    detections.save_frame(
        run_id,
        _tracked_frame(track_id=1, sample_index=2, timestamp_seconds=2),
    )
    second = resolver.resolve(
        faces.save_event(
            run_id,
            _event(
                track_id=2,
                sample_index=20,
                timestamp_seconds=10,
                matched=False,
                similarity=0.1,
                x=180,
                y=100,
                width=30,
                height=30,
            ),
        )
    )

    assert first is not None
    assert second is not None
    assert first.id != second.id


def test_resolver_upgrades_unknown_result_on_the_same_track(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    run_id = _run(database_path)
    faces = FaceMatchRepository(database_path)
    people = PersonInstanceRepository(database_path)
    resolver = TrackIdentityResolver(people)

    first = resolver.resolve(
        faces.save_event(
            run_id,
            _event(
                track_id=7,
                sample_index=1,
                timestamp_seconds=1,
                matched=False,
                similarity=0.2,
                x=10,
                y=10,
                width=20,
                height=20,
            ),
        )
    )
    second = resolver.resolve(
        faces.save_event(
            run_id,
            _event(
                track_id=7,
                sample_index=5,
                timestamp_seconds=3,
                matched=True,
                similarity=0.8,
                x=20,
                y=10,
                width=20,
                height=20,
            ),
        )
    )

    assert first is not None
    assert second is not None
    assert first.id == second.id
    assert second.status == "matched"
    assert second.track_count == 1
    assert people.summarize_run(run_id).unknown_instances == 0


def test_person_instance_api_returns_final_people_and_linked_tracks(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=database_path,
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "cctv.jsonl",
    )

    with TestClient(create_app(settings)) as client:
        run_id = _run(database_path)
        faces = FaceMatchRepository(database_path)
        people = PersonInstanceRepository(database_path)
        resolved = TrackIdentityResolver(people).resolve(
            faces.save_event(
                run_id,
                _event(
                    track_id=7,
                    sample_index=3,
                    timestamp_seconds=1.5,
                    matched=True,
                    similarity=0.8,
                    x=10,
                    y=10,
                    width=20,
                    height=20,
                ),
            )
        )
        assert resolved is not None

        instances = client.get(
            "/person-instances",
            params={"analysis_run_id": run_id, "status": "matched"},
        )
        summary = client.get(f"/analysis-runs/{run_id}/person-instance-summary")
        links = client.get(
            f"/analysis-runs/{run_id}/person-instances/{resolved.id}/tracks"
        )
        missing = client.get(
            f"/analysis-runs/{run_id}/person-instances/missing/tracks"
        )

    assert instances.status_code == 200
    assert instances.json()["total"] == 1
    assert instances.json()["items"][0]["id"] == resolved.id
    assert instances.json()["items"][0]["identity_id"] == "emp-001"
    assert summary.json() == {
        "analysis_run_id": run_id,
        "total_instances": 1,
        "matched_instances": 1,
        "unknown_instances": 0,
        "linked_tracks": 1,
    }
    assert links.status_code == 200
    assert links.json()["items"][0]["track_id"] == 7
    assert links.json()["items"][0]["linked_by"] == "face_match"
    assert missing.status_code == 404
