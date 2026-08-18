import json
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from cctv.core.settings import get_settings
from cctv.db import AnalysisRunStatus, DetectionRepository
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


def test_rtsp_cli_persists_yolo_results_without_exposing_url(
    monkeypatch,
    tmp_path: Path,
    capsys,
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
    monkeypatch.setenv("CCTV_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("CCTV_LOG_PATH", str(tmp_path / "runtime" / "cctv.jsonl"))
    get_settings.cache_clear()

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
    assert output["source_name"] == "camera-1"
    assert output["analysis"]["analysis_run_id"] == runs.items[0].id
    assert runs.total == 1
    assert runs.items[0].source_type == "rtsp"
    assert runs.items[0].source_name == "camera-1"
    assert runs.items[0].status is AnalysisRunStatus.COMPLETED
    assert runs.items[0].processed_frames == 2
    assert repository.list_detections(analysis_run_id=runs.items[0].id).total == 2
