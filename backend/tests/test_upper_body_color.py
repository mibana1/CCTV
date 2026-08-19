import cv2
import numpy as np

from cctv.inference import BoundingBox, Detection, FrameDetections
from cctv.media import DecodedFrame
from cctv.vision import UpperBodyColorAnalyzer


def _analyze_color(bgr: tuple[int, int, int]):
    image = np.full((120, 100, 3), bgr, dtype=np.uint8)
    frame = DecodedFrame(
        source_index=1,
        sample_index=1,
        timestamp_seconds=0.5,
        image=image,
    )
    result = FrameDetections(
        source_index=1,
        sample_index=1,
        timestamp_seconds=0.5,
        frame_width=100,
        frame_height=120,
        inference_seconds=0.01,
        detections=(
            Detection(
                class_id=0,
                label="person",
                confidence=0.9,
                box=BoundingBox(10, 5, 90, 115),
                track_id=7,
            ),
        ),
    )
    analyzer = UpperBodyColorAnalyzer()
    return analyzer, analyzer.analyze(frame, result)[7]


def test_upper_body_color_analyzer_classifies_primary_and_achromatic_colors() -> None:
    _, red = _analyze_color((0, 0, 255))
    _, blue = _analyze_color((255, 0, 0))
    _, black = _analyze_color((20, 20, 20))
    _, white = _analyze_color((240, 240, 240))

    assert red.color == "red"
    assert blue.color == "blue"
    assert black.color == "black"
    assert white.color == "white"
    assert all(item.confidence == 1 for item in (red, blue, black, white))
    assert red.rule_attributes["upper_body_crop_box"] == [22, 25, 78, 73]


def test_upper_body_color_analyzer_uses_crop_center_and_reports_distribution() -> None:
    image = np.full((120, 100, 3), (255, 0, 0), dtype=np.uint8)
    cv2.rectangle(image, (35, 30), (65, 70), (0, 0, 255), thickness=-1)
    frame = DecodedFrame(0, 0, 0, image)
    result = FrameDetections(
        source_index=0,
        sample_index=0,
        timestamp_seconds=0,
        frame_width=100,
        frame_height=120,
        inference_seconds=0.01,
        detections=(Detection(0, "person", 0.9, BoundingBox(10, 5, 90, 115), 1),),
    )

    observation = UpperBodyColorAnalyzer().analyze(frame, result)[1]

    assert observation.color == "red"
    assert observation.distribution["red"] > observation.distribution["blue"]
    assert 0.5 < observation.confidence < 1


def test_upper_body_color_analyzer_skips_tiny_or_untracked_people() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    frame = DecodedFrame(0, 0, 0, image)
    result = FrameDetections(
        source_index=0,
        sample_index=0,
        timestamp_seconds=0,
        frame_width=20,
        frame_height=20,
        inference_seconds=0.01,
        detections=(
            Detection(0, "person", 0.9, BoundingBox(0, 0, 5, 5), 1),
            Detection(0, "person", 0.9, BoundingBox(0, 0, 20, 20)),
        ),
    )
    analyzer = UpperBodyColorAnalyzer()

    assert analyzer.analyze(frame, result) == {}
    assert analyzer.summary.person_detections == 1
    assert analyzer.summary.skipped_crops == 1
