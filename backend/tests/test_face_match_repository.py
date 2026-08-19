from pathlib import Path

from cctv.db import (
    AnalysisRunStatus,
    DetectionRepository,
    FaceMatchEventInput,
    FaceMatchRepository,
    initialize_database,
)


def _event(*, matched: bool, sample_index: int) -> FaceMatchEventInput:
    return FaceMatchEventInput(
        source_index=sample_index * 10,
        sample_index=sample_index,
        source_timestamp_seconds=sample_index * 0.5,
        face_index=0,
        track_id=sample_index + 1,
        face_x=10,
        face_y=20,
        face_width=30,
        face_height=40,
        detection_confidence=0.9,
        match_status="matched" if matched else "unknown",
        rejection_reason=None if matched else "below_threshold",
        identity_id="emp-001" if matched else None,
        external_id="EMP-001" if matched else None,
        display_name="Test Person" if matched else None,
        best_candidate_identity_id="emp-001",
        best_candidate_external_id="EMP-001",
        best_similarity=0.62 if matched else 0.32,
        second_best_similarity=None,
        similarity_threshold=0.45,
        minimum_margin=0.05,
    )


def test_face_match_repository_persists_filters_and_summarizes_events(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    detections = DetectionRepository(database_path)
    run = detections.create_analysis_run(
        source_type="rtsp",
        source_name="camera-1",
        model_name="model.onnx",
        model_sha256="a" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
    )
    repository = FaceMatchRepository(database_path)

    matched = repository.save_event(run.id, _event(matched=True, sample_index=1))
    unknown = repository.save_event(run.id, _event(matched=False, sample_index=2))
    detections.finish_analysis_run(run.id, status=AnalysisRunStatus.COMPLETED)

    page = repository.list_events(analysis_run_id=run.id)
    matched_page = repository.list_events(analysis_run_id=run.id, match_status="matched")
    summary = repository.summarize_run(run.id)

    assert page.total == 2
    assert [item.id for item in page.items] == [unknown.id, matched.id]
    assert matched_page.total == 1
    assert matched_page.items[0].identity_id == "emp-001"
    assert matched_page.items[0].track_id == 2
    assert summary.detected_faces == 2
    assert summary.matched_faces == 1
    assert summary.unknown_faces == 1
    assert summary.best_similarity == 0.62

