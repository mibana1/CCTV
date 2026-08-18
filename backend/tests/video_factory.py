from pathlib import Path

import cv2
import numpy as np
import pytest


def create_test_video(path: Path, *, fps: float = 4.0, frame_count: int = 12) -> Path:
    """Create a small deterministic MJPG video for media and worker tests."""
    width, height = 32, 24
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.fail("OpenCV MJPG test writer could not be opened")

    try:
        for index in range(frame_count):
            image = np.full((height, width, 3), index * 10, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()

    return path
