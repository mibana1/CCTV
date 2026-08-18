import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cctv.core.settings import get_settings
from cctv.db import AnalysisRunStatus, DetectionRepository
from cctv.media import DecodedFrame, SnapshotWriter
from cctv.workers import (
    LocalVideoWorker,
    SequentialFrameConsumer,
    WorkerStatus,
    WorkerStopReason,
)
from cctv.workers.local_video import run as run_local_video_worker
from tests.video_factory import create_test_video


def test_local_video_worker_dispatches_sampled_frames(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")
    consumed: list[tuple[int, int, float]] = []

    def consume(frame: DecodedFrame) -> None:
        consumed.append((frame.source_index, frame.sample_index, frame.timestamp_seconds))

    worker = LocalVideoWorker(video_path, sample_fps=2, frame_consumer=consume)
    result = worker.execute()

    assert worker.status is WorkerStatus.COMPLETED
    assert result.status is WorkerStatus.COMPLETED
    assert result.stop_reason is WorkerStopReason.END_OF_STREAM
    assert result.processed_samples == 6
    assert result.first_source_index == 0
    assert result.last_source_index == 10
    assert result.first_timestamp_seconds == pytest.approx(0)
    assert result.last_timestamp_seconds == pytest.approx(2.5)
    assert result.metadata is not None
    assert result.metadata.frame_count == 12
    assert consumed == pytest.approx(
        [(0, 0, 0), (2, 1, 0.5), (4, 2, 1), (6, 3, 1.5), (8, 4, 2), (10, 5, 2.5)]
    )


def test_local_video_worker_honors_sample_limit(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")
    worker = LocalVideoWorker(video_path, sample_fps=2, max_samples=2)

    result = worker.execute()

    assert result.status is WorkerStatus.COMPLETED
    assert result.stop_reason is WorkerStopReason.SAMPLE_LIMIT
    assert result.processed_samples == 2
    assert result.last_source_index == 2


def test_local_video_worker_saves_samples_through_snapshot_consumer(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")
    snapshot_writer = SnapshotWriter(tmp_path / "snapshots")
    worker = LocalVideoWorker(
        video_path,
        sample_fps=2,
        frame_consumer=snapshot_writer,
        max_samples=2,
    )

    result = worker.execute()

    snapshots = sorted(snapshot_writer.output_dir.glob("*.jpg"))
    assert result.processed_samples == 2
    assert snapshot_writer.saved_count == 2
    assert [path.name for path in snapshots] == [
        "sample_000000_frame_000000000_t000000000000ms.jpg",
        "sample_000001_frame_000000002_t000000000500ms.jpg",
    ]
    assert all(cv2.imread(str(path)) is not None for path in snapshots)


def test_sequential_frame_consumer_dispatches_in_order() -> None:
    calls: list[tuple[str, int]] = []
    frame = DecodedFrame(
        source_index=3,
        sample_index=1,
        timestamp_seconds=0.5,
        image=cv2.UMat(2, 2, cv2.CV_8UC3).get(),
    )
    consumer = SequentialFrameConsumer(
        lambda value: calls.append(("analysis", value.source_index)),
        lambda value: calls.append(("snapshot", value.source_index)),
    )

    consumer(frame)

    assert calls == [("analysis", 3), ("snapshot", 3)]


def test_local_video_worker_honors_consumer_stop_request(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")

    def consume(frame: DecodedFrame) -> None:
        del frame
        worker.stop()

    worker = LocalVideoWorker(video_path, sample_fps=2, frame_consumer=consume)
    result = worker.execute()

    assert result.status is WorkerStatus.STOPPED
    assert result.stop_reason is WorkerStopReason.STOP_REQUESTED
    assert result.processed_samples == 1


def test_local_video_worker_marks_consumer_failure(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")

    def fail(frame: DecodedFrame) -> None:
        del frame
        raise RuntimeError("consumer failed")

    worker = LocalVideoWorker(video_path, sample_fps=2, frame_consumer=fail)

    with pytest.raises(RuntimeError, match="consumer failed"):
        worker.execute()

    assert worker.status is WorkerStatus.FAILED


def test_local_video_worker_is_single_use(tmp_path: Path) -> None:
    video_path = create_test_video(tmp_path / "sample.avi")
    worker = LocalVideoWorker(video_path, sample_fps=2, max_samples=1)
    worker.execute()

    with pytest.raises(RuntimeError, match="only be executed once"):
        worker.execute()


def test_local_video_cli_persists_yolo_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeNetwork:
        def __init__(self) -> None:
            self.output = np.zeros((1, 84, 1), dtype=np.float32)
            self.output[0, :4, 0] = [320, 320, 100, 100]
            self.output[0, 4, 0] = 0.9

        def setPreferableBackend(self, backend_id: int) -> None:
            del backend_id

        def setInput(self, blob: np.ndarray) -> None:
            del blob

        def forward(self) -> np.ndarray:
            return self.output

    video_path = create_test_video(tmp_path / "sample.avi")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test model placeholder")
    database_path = tmp_path / "runtime" / "cctv.db"
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: FakeNetwork())
    monkeypatch.setenv("CCTV_APP_MODE", "dry_run")
    monkeypatch.setenv("CCTV_AI_DEVICE", "cpu")
    monkeypatch.setenv("CCTV_YOLO_ENABLED", "false")
    monkeypatch.setenv("CCTV_PERSIST_DETECTIONS", "true")
    monkeypatch.setenv("CCTV_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CCTV_LOG_PATH", str(tmp_path / "runtime" / "cctv.jsonl"))
    get_settings.cache_clear()

    try:
        run_local_video_worker(
            [
                str(video_path),
                "--analyze-yolo",
                "--model-path",
                str(model_path),
                "--sample-fps",
                "2",
                "--max-samples",
                "2",
            ]
        )
    finally:
        get_settings.cache_clear()

    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    repository = DetectionRepository(database_path)
    runs = repository.list_analysis_runs()

    assert output["analysis"]["persistence_enabled"] is True
    assert output["analysis"]["analysis_run_id"] == runs.items[0].id
    assert runs.total == 1
    assert runs.items[0].status is AnalysisRunStatus.COMPLETED
    assert runs.items[0].processed_frames == 2
    assert runs.items[0].total_detections == 2
    assert repository.list_detections(analysis_run_id=runs.items[0].id).total == 2


@pytest.mark.parametrize("max_samples", [0, -1])
def test_local_video_worker_rejects_invalid_sample_limit(max_samples: int) -> None:
    with pytest.raises(ValueError, match="max_samples"):
        LocalVideoWorker("sample.mp4", sample_fps=2, max_samples=max_samples)


@pytest.mark.parametrize("sample_fps", [0, -1, float("inf"), float("nan")])
def test_local_video_worker_rejects_invalid_sample_fps(sample_fps: float) -> None:
    with pytest.raises(ValueError, match="sample_fps"):
        LocalVideoWorker("sample.mp4", sample_fps=sample_fps)
