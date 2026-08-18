from pathlib import Path

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.db import AnalysisRunStatus, DetectionRepository
from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.main import create_app


def create_result(
    sample_index: int,
    class_name: str,
    confidence: float,
    track_id: int | None = None,
) -> FrameDetections:
    return FrameDetections(
        source_index=sample_index * 2,
        sample_index=sample_index,
        timestamp_seconds=sample_index * 0.5,
        frame_width=1920,
        frame_height=1080,
        inference_seconds=0.04,
        detections=(
            Detection(
                class_id=0 if class_name == "person" else 2,
                label=class_name,
                confidence=confidence,
                box=BoundingBox(x1=100, y1=200, x2=300, y2=500),
                track_id=track_id,
            ),
        ),
    )


def create_run(repository: DetectionRepository, run_id: str, source_name: str) -> str:
    run = repository.create_analysis_run(
        source_type="local_video",
        source_name=source_name,
        model_name="model.onnx",
        model_sha256="b" * 64,
        device="cpu",
        input_size=640,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        sample_fps=2,
        run_id=run_id,
    )
    return run.id


def test_analysis_api_filters_results_and_enforces_pagination(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "cctv.db"
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_path=database_path,
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "runtime" / "cctv.jsonl",
    )

    with TestClient(create_app(settings)) as client:
        repository = DetectionRepository(database_path)
        first_run_id = create_run(repository, "run-1", "first.mp4")
        repository.save_frame(first_run_id, create_result(0, "person", 0.9, track_id=1))
        repository.save_frame(first_run_id, create_result(1, "person", 0.4, track_id=1))
        repository.save_frame(first_run_id, create_result(2, "car", 0.8, track_id=2))
        repository.finish_analysis_run(first_run_id, status=AnalysisRunStatus.COMPLETED)

        second_run_id = create_run(repository, "run-2", "second.mp4")
        repository.save_frame(second_run_id, create_result(0, "person", 0.95))
        repository.finish_analysis_run(second_run_id, status=AnalysisRunStatus.FAILED)

        run_page = client.get("/analysis-runs", params={"page": 1, "limit": 1})
        completed_runs = client.get("/analysis-runs", params={"status": "completed"})
        run_detail = client.get(f"/analysis-runs/{first_run_id}")
        detections = client.get(
            "/detections",
            params={
                "analysis_run_id": first_run_id,
                "class_name": "PERSON",
                "min_confidence": 0.5,
                "limit": 10,
            },
        )
        second_detection_page = client.get("/detections", params={"page": 2, "limit": 2})
        tracked_detections = client.get(
            "/detections",
            params={"analysis_run_id": first_run_id, "track_id": 1},
        )
        tracks = client.get(
            "/tracks",
            params={
                "analysis_run_id": first_run_id,
                "class_name": "PERSON",
                "min_observations": 2,
                "observed_from_seconds": 0.4,
                "observed_to_seconds": 0.6,
                "active": "false",
            },
        )
        active_tracks = client.get(
            "/tracks",
            params={"analysis_run_id": first_run_id, "active": "true"},
        )
        track_detail = client.get(f"/analysis-runs/{first_run_id}/tracks/1")
        observations = client.get(
            f"/analysis-runs/{first_run_id}/tracks/1/observations",
            params={"page": 1, "limit": 1},
        )

        assert run_page.status_code == 200
        assert run_page.json()["total"] == 2
        assert run_page.json()["has_next"] is True
        assert len(run_page.json()["items"]) == 1
        assert completed_runs.json()["total"] == 1
        assert completed_runs.json()["items"][0]["id"] == first_run_id

        assert run_detail.status_code == 200
        assert run_detail.json()["processed_frames"] == 3
        assert run_detail.json()["total_detections"] == 3
        assert run_detail.json()["source_name"] == "first.mp4"
        assert run_detail.json()["detector_type"] == "yolo_onnx"
        assert "model_path" not in run_detail.json()

        assert detections.status_code == 200
        assert detections.json()["total"] == 1
        assert detections.json()["items"][0]["class_name"] == "person"
        assert detections.json()["items"][0]["confidence"] == 0.9
        assert detections.json()["items"][0]["track_id"] == 1
        assert detections.json()["items"][0]["box"] == {
            "x1": 100,
            "y1": 200,
            "x2": 300,
            "y2": 500,
        }
        assert second_detection_page.status_code == 200
        assert second_detection_page.json()["total"] == 4
        assert len(second_detection_page.json()["items"]) == 2
        assert tracked_detections.status_code == 200
        assert tracked_detections.json()["total"] == 2
        assert {item["track_id"] for item in tracked_detections.json()["items"]} == {1}
        assert tracks.status_code == 200
        assert tracks.json()["total"] == 1
        assert tracks.json()["items"][0]["track_id"] == 1
        assert tracks.json()["items"][0]["observation_count"] == 2
        assert tracks.json()["items"][0]["is_active"] is False
        assert active_tracks.status_code == 200
        assert active_tracks.json()["total"] == 0
        assert track_detail.status_code == 200
        assert track_detail.json()["first_sample_index"] == 0
        assert track_detail.json()["last_sample_index"] == 1
        assert track_detail.json()["max_confidence"] == 0.9
        assert observations.status_code == 200
        assert observations.json()["total"] == 2
        assert observations.json()["has_next"] is True
        assert observations.json()["items"][0]["sample_index"] == 0
        assert observations.json()["items"][0]["box"] == {
            "x1": 100,
            "y1": 200,
            "x2": 300,
            "y2": 500,
        }

        assert client.get("/analysis-runs/missing").status_code == 404
        assert client.get("/analysis-runs", params={"limit": 101}).status_code == 422
        assert client.get("/detections", params={"page": 0}).status_code == 422
        assert client.get("/detections", params={"min_confidence": 1.1}).status_code == 422
        assert client.get("/detections", params={"track_id": 1}).status_code == 422
        assert client.get("/tracks", params={"min_observations": 0}).status_code == 422
        assert (
            client.get(
                "/tracks",
                params={"observed_from_seconds": 2, "observed_to_seconds": 1},
            ).status_code
            == 422
        )
        assert client.get("/tracks", params={"observed_from_seconds": -1}).status_code == 422
        assert client.get(f"/analysis-runs/{first_run_id}/tracks/0").status_code == 422
        assert client.get(f"/analysis-runs/{first_run_id}/tracks/999").status_code == 404
        assert (
            client.get(f"/analysis-runs/{first_run_id}/tracks/999/observations").status_code == 404
        )
