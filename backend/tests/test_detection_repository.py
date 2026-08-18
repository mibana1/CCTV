import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import (
    AnalysisRunStatus,
    DetectionRepository,
    connect_database,
    initialize_database,
)
from cctv.inference import BoundingBox, Detection, FrameDetections


def make_result(
    sample_index: int,
    *detections: Detection,
) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index * 2,
        sample_index=sample_index,
        timestamp_seconds=sample_index * 0.5,
        frame_width=640,
        frame_height=360,
        inference_seconds=0.025,
        detections=detections,
    )


def make_detection(
    class_id: int,
    class_name: str,
    confidence: float,
    track_id: int | None = None,
) -> Detection:
    return Detection(
        class_id=class_id,
        label=class_name,
        confidence=confidence,
        box=BoundingBox(x1=10, y1=20, x2=110, y2=220),
        track_id=track_id,
    )


def create_run(repository: DetectionRepository, run_id: str = "run-1"):
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
        run_id=run_id,
    )


def test_repository_persists_frames_detections_and_run_counts(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = DetectionRepository(database_path)
    run = create_run(repository)

    first_frame_id = repository.save_frame(
        run.id,
        make_result(
            0,
            make_detection(0, "person", 0.9, track_id=1),
            make_detection(2, "car", 0.6),
        ),
    )
    repository.save_frame(run.id, make_result(1))
    completed = repository.finish_analysis_run(
        run.id,
        status=AnalysisRunStatus.COMPLETED,
    )

    assert first_frame_id > 0
    assert completed.status is AnalysisRunStatus.COMPLETED
    assert completed.completed_at is not None
    assert completed.processed_frames == 2
    assert completed.total_detections == 2
    assert completed.error_type is None

    detections = repository.list_detections(analysis_run_id=run.id)
    assert detections.total == 2
    assert [item.class_name for item in detections.items] == ["person", "car"]
    assert detections.items[0].frame_width == 640
    assert detections.items[0].frame_height == 360
    assert detections.items[0].track_id == 1
    assert detections.items[0].x2 == 110

    with closing(connect_database(database_path)) as connection:
        frames = connection.execute(
            """
            SELECT sample_index, detection_count
            FROM analyzed_frames
            WHERE analysis_run_id = ?
            ORDER BY sample_index
            """,
            (run.id,),
        ).fetchall()
    assert [tuple(frame) for frame in frames] == [(0, 2), (1, 0)]


def test_repository_filters_and_paginates_detections(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = DetectionRepository(database_path)
    first_run = create_run(repository, "run-1")
    repository.save_frame(
        first_run.id,
        make_result(
            0,
            make_detection(0, "person", 0.9, track_id=1),
            make_detection(0, "person", 0.4),
            make_detection(2, "car", 0.8),
        ),
    )
    repository.finish_analysis_run(first_run.id, status=AnalysisRunStatus.COMPLETED)
    second_run = create_run(repository, "run-2")
    repository.save_frame(second_run.id, make_result(0, make_detection(0, "person", 0.95)))
    repository.finish_analysis_run(
        second_run.id, status=AnalysisRunStatus.FAILED, error_type="TestError"
    )

    first_page = repository.list_detections(page=1, limit=2)
    second_page = repository.list_detections(page=2, limit=2)
    filtered = repository.list_detections(
        analysis_run_id=first_run.id,
        class_name="PERSON",
        min_confidence=0.5,
    )
    tracked = repository.list_detections(analysis_run_id=first_run.id, track_id=1)
    failed_runs = repository.list_analysis_runs(status=AnalysisRunStatus.FAILED)

    assert first_page.total == 4
    assert len(first_page.items) == 2
    assert len(second_page.items) == 2
    assert filtered.total == 1
    assert filtered.items[0].confidence == pytest.approx(0.9)
    assert tracked.total == 1
    assert tracked.items[0].track_id == 1
    assert failed_runs.total == 1
    assert failed_runs.items[0].id == second_run.id
    assert failed_runs.items[0].error_type == "TestError"


def test_repository_rolls_back_duplicate_frame_and_rejects_finished_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = DetectionRepository(database_path)
    run = create_run(repository)
    result = make_result(0, make_detection(0, "person", 0.9))
    repository.save_frame(run.id, result)

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_frame(run.id, result)

    repository.finish_analysis_run(run.id, status=AnalysisRunStatus.COMPLETED)
    with pytest.raises(RuntimeError, match="running analysis"):
        repository.save_frame(run.id, make_result(1))

    with closing(connect_database(database_path)) as connection:
        frame_count = connection.execute("SELECT COUNT(*) FROM analyzed_frames").fetchone()[0]
        detection_count = connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    assert frame_count == 1
    assert detection_count == 1


@pytest.mark.parametrize(
    ("page", "limit"),
    [(0, 10), (1, 0), (1, 101)],
)
def test_repository_rejects_invalid_pagination(tmp_path: Path, page: int, limit: int) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = DetectionRepository(database_path)

    with pytest.raises(ValueError):
        repository.list_detections(page=page, limit=limit)


def test_repository_requires_run_when_filtering_by_track(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = DetectionRepository(database_path)

    with pytest.raises(ValueError, match="analysis_run_id is required"):
        repository.list_detections(track_id=1)
