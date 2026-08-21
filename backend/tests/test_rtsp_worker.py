import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from cctv.core.settings import get_settings
from cctv.db import (
    AnalysisRunStatus,
    DetectionRepository,
    DisplayActionRepository,
    RuleRepository,
    initialize_database,
)
from cctv.identity import FaceMatchingRunSummary
from cctv.media import RawRtspFrame, RtspIngestStatistics
from cctv.workers import RtspStopReason, RtspWorker, WorkerStatus
from cctv.workers import rtsp as rtsp_worker_module


class FakeReader:
    def __init__(self, timestamps: tuple[float, ...], *, source_name: str = "camera") -> None:
        self.timestamps = timestamps
        self.source_name = source_name

    @property
    def statistics(self) -> RtspIngestStatistics:
        return RtspIngestStatistics(
            connection_count=1,
            reconnect_count=0,
            decoded_frames=len(self.timestamps),
        )

    def frames(self, *, stop_event: Event | None = None) -> Iterator[RawRtspFrame]:
        for index, timestamp in enumerate(self.timestamps):
            if stop_event is not None and stop_event.is_set():
                break
            yield RawRtspFrame(
                source_index=index,
                timestamp_seconds=timestamp,
                image=np.full((24, 32, 3), index, dtype=np.uint8),
            )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_rtsp_worker_samples_by_elapsed_stream_time() -> None:
    consumed: list[tuple[int, int, float]] = []
    reader = FakeReader((0, 0.1, 0.49, 0.5, 0.9, 1.0))
    worker = RtspWorker(
        reader,  # type: ignore[arg-type]
        sample_fps=2,
        max_samples=3,
        frame_consumer=lambda frame: consumed.append(
            (frame.source_index, frame.sample_index, frame.timestamp_seconds)
        ),
    )

    result = worker.execute()

    assert result.status is WorkerStatus.COMPLETED
    assert result.stop_reason is RtspStopReason.SAMPLE_LIMIT
    assert result.processed_samples == 3
    assert result.decoded_frames == 6
    assert consumed == [(0, 0, 0), (3, 1, 0.5), (5, 2, 1.0)]


def test_rtsp_worker_emits_five_second_progress_heartbeats_and_final_state(caplog) -> None:
    clock = FakeClock()
    last_frame_at = datetime(2026, 8, 21, 1, 2, 3, 456000, tzinfo=UTC)
    worker = RtspWorker(
        FakeReader((0, 1, 2, 3)),  # type: ignore[arg-type]
        sample_fps=1,
        max_samples=4,
        frame_consumer=lambda frame: clock.advance(2),
        emit_progress=True,
        monotonic_clock=clock,
        now_factory=lambda: last_frame_at,
    )
    caplog.set_level("INFO", logger=rtsp_worker_module.__name__)

    worker.execute()

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "rtsp_worker_progress"
    ]
    assert [record.reason for record in records] == ["started", "interval", "finished"]
    assert [record.processed_samples for record in records] == [0, 3, 4]
    assert [record.interval_samples for record in records] == [0, 3, 1]
    assert [record.interval_fps for record in records] == [0.0, 0.5, 0.5]
    assert records[0].last_frame_at is None
    assert records[1].last_frame_at == "2026-08-21T01:02:03.456Z"
    assert records[2].last_frame_at == "2026-08-21T01:02:03.456Z"
    assert all(record.progress_interval_seconds == 5.0 for record in records)


def test_rtsp_worker_emits_error_immediately_and_forces_failed_final_progress(caplog) -> None:
    def fail_consumer(frame) -> None:
        del frame
        raise RuntimeError("failed")

    worker = RtspWorker(
        FakeReader((0,)),  # type: ignore[arg-type]
        sample_fps=1,
        frame_consumer=fail_consumer,
        emit_progress=True,
    )
    caplog.set_level("INFO", logger=rtsp_worker_module.__name__)

    with pytest.raises(RuntimeError, match="failed"):
        worker.execute()

    worker_records = [
        record
        for record in caplog.records
        if getattr(record, "event", "").startswith("rtsp_worker_")
    ]
    failed_index = next(
        index
        for index, record in enumerate(worker_records)
        if record.event == "rtsp_worker_failed"
    )
    final_progress = worker_records[-1]
    assert final_progress.event == "rtsp_worker_progress"
    assert final_progress.reason == "finished"
    assert final_progress.status is WorkerStatus.FAILED
    assert failed_index == len(worker_records) - 2
    assert worker_records[failed_index].error_type == "RuntimeError"


@pytest.mark.parametrize(
    ("frame_persistence_mode", "expected_persisted_frames"),
    [("all", 2), ("detections", 2), ("events", 1)],
)
def test_rtsp_cli_persists_yolo_results_by_frame_policy_without_exposing_url(
    monkeypatch,
    tmp_path: Path,
    capsys,
    frame_persistence_mode: str,
    expected_persisted_frames: int,
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

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test model placeholder")
    database_path = tmp_path / "runtime" / "cctv.db"
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: FakeNetwork())
    monkeypatch.setattr(
        rtsp_worker_module,
        "RtspStreamReader",
        lambda *args, **kwargs: FakeReader((0, 0.5), source_name=kwargs["source_name"]),
    )
    monkeypatch.setenv("CCTV_APP_MODE", "dry_run")
    monkeypatch.setenv("CCTV_AI_DEVICE", "cpu")
    monkeypatch.setenv("CCTV_YOLO_ENABLED", "false")
    monkeypatch.setenv("CCTV_PERSIST_DETECTIONS", "true")
    monkeypatch.setenv("CCTV_FRAME_PERSISTENCE_MODE", frame_persistence_mode)
    monkeypatch.setenv("CCTV_RULES_ENABLED", "true")
    monkeypatch.setenv("CCTV_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CCTV_LOG_PATH", str(tmp_path / "runtime" / "cctv.jsonl"))
    get_settings.cache_clear()
    initialize_database(database_path)
    rule_repository = RuleRepository(database_path)
    rule_repository.create_rule(
        name="Entire frame",
        source_name="camera-1",
        rule_type="intrusion",
        class_name="person",
        geometry={
            "type": "polygon",
            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
    )

    try:
        rtsp_worker_module.run(
            [
                "--rtsp-url",
                "rtsp://secret-user:secret-password@camera.local/stream",
                "--source-name",
                "camera-1",
                "--analyze-yolo",
                "--model-path",
                str(model_path),
                "--sample-fps",
                "2",
                "--max-samples",
                "2",
                "--emit-progress",
            ]
        )
    finally:
        get_settings.cache_clear()

    captured_output = capsys.readouterr().out
    output = json.loads(captured_output.splitlines()[-1])
    repository = DetectionRepository(database_path)
    runs = repository.list_analysis_runs()

    assert "secret-user" not in captured_output
    assert "secret-password" not in captured_output
    assert '"event":"rtsp_worker_progress"' in captured_output
    assert output["source_name"] == "camera-1"
    assert output["analysis"]["frame_persistence_mode"] == frame_persistence_mode
    assert output["analysis"]["analysis_run_id"] == runs.items[0].id
    assert runs.total == 1
    assert runs.items[0].source_type == "rtsp"
    assert runs.items[0].source_name == "camera-1"
    assert runs.items[0].status is AnalysisRunStatus.COMPLETED
    assert runs.items[0].processed_frames == expected_persisted_frames
    assert (
        repository.list_detections(analysis_run_id=runs.items[0].id).total
        == expected_persisted_frames
    )
    assert rule_repository.list_events(analysis_run_id=runs.items[0].id).total == 1
    assert DisplayActionRepository(database_path).list(
        analysis_run_id=runs.items[0].id
    ).total == 1


def test_rtsp_cli_reuses_face_matching_consumer(monkeypatch, tmp_path: Path, capsys) -> None:
    class FakeFaceMatchingConsumer:
        def __init__(self) -> None:
            self.processed_frames = 0

        def __call__(self, frame) -> None:
            del frame
            self.processed_frames += 1

        @property
        def summary(self) -> FaceMatchingRunSummary:
            return FaceMatchingRunSummary(
                candidate_identity_count=1,
                candidate_embedding_count=3,
                similarity_threshold=0.45,
                minimum_margin=0.05,
                processed_frames=self.processed_frames,
                frames_with_faces=0,
                detected_faces=0,
                matched_faces=0,
                unknown_faces=0,
                best_similarity=None,
                matched_identity_counts={},
            )

    face_consumer = FakeFaceMatchingConsumer()
    monkeypatch.setattr(
        rtsp_worker_module,
        "RtspStreamReader",
        lambda *args, **kwargs: FakeReader((0, 0.5), source_name=kwargs["source_name"]),
    )
    monkeypatch.setattr(
        rtsp_worker_module,
        "create_sface_matching_consumer",
        lambda *args, **kwargs: face_consumer,
    )
    monkeypatch.setenv("CCTV_DETECTOR_ENABLED", "false")
    monkeypatch.setenv("CCTV_YOLO_ENABLED", "false")
    monkeypatch.setenv("CCTV_FACE_MATCHING_ENABLED", "false")
    monkeypatch.setenv("CCTV_LOG_PATH", str(tmp_path / "cctv.jsonl"))
    get_settings.cache_clear()

    try:
        rtsp_worker_module.run(
            ["--match-faces", "--sample-fps", "2", "--max-samples", "2"]
        )
    finally:
        get_settings.cache_clear()

    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert face_consumer.processed_frames == 2
    assert output["face_matching"]["candidate_embedding_count"] == 3
