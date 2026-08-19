"""Render database-backed tracking overlays without modifying source snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cctv.media.snapshot import SnapshotEncodingError


@dataclass(frozen=True, slots=True)
class SnapshotOverlay:
    """One rectangular overlay and its optional second line of identity text."""

    x1: int
    y1: int
    x2: int
    y2: int
    primary_label: str
    secondary_label: str | None = None
    color: tuple[int, int, int] = (255, 140, 0)


def render_snapshot_annotations(
    snapshot_path: str | Path,
    overlays: tuple[SnapshotOverlay, ...],
    *,
    jpeg_quality: int = 90,
) -> bytes:
    """Decode a source JPEG, draw current tracking labels, and return a new JPEG."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    path = Path(snapshot_path).expanduser().resolve()
    encoded_source = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded_source, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SnapshotEncodingError(f"snapshot could not be decoded: {path.name}")

    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.45, min(0.8, max(width, height) / 1_200))
    line_thickness = max(1, round(scale * 2))
    box_thickness = max(2, round(scale * 3))
    padding = max(3, round(scale * 6))

    for overlay in overlays:
        x1 = min(max(0, overlay.x1), width - 1)
        y1 = min(max(0, overlay.y1), height - 1)
        x2 = min(max(x1 + 1, overlay.x2), width - 1)
        y2 = min(max(y1 + 1, overlay.y2), height - 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), overlay.color, box_thickness)
        _draw_label(
            canvas,
            x1=x1,
            y1=y1,
            y2=y2,
            labels=tuple(
                label
                for label in (overlay.primary_label, overlay.secondary_label)
                if label
            ),
            color=overlay.color,
            scale=scale,
            thickness=line_thickness,
            padding=padding,
        )

    encode_ok, encoded_result = cv2.imencode(
        ".jpg",
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not encode_ok:
        raise SnapshotEncodingError(f"annotated snapshot encoding failed: {path.name}")
    return encoded_result.tobytes()


def _draw_label(
    image: np.ndarray,
    *,
    x1: int,
    y1: int,
    y2: int,
    labels: tuple[str, ...],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
    padding: int,
) -> None:
    if not labels:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(label, font, scale, thickness)[0] for label in labels]
    line_height = max(size[1] for size in sizes) + padding
    block_height = line_height * len(labels) + padding
    block_width = max(size[0] for size in sizes) + padding * 2
    image_height, image_width = image.shape[:2]
    label_x1 = min(x1, max(0, image_width - block_width))
    label_x2 = min(image_width - 1, label_x1 + block_width)
    if y1 >= block_height:
        label_y1 = y1 - block_height
        label_y2 = y1
    else:
        label_y1 = min(image_height - block_height, y2)
        label_y1 = max(0, label_y1)
        label_y2 = min(image_height - 1, label_y1 + block_height)
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    for index, label in enumerate(labels):
        baseline_y = label_y1 + padding + sizes[index][1] + index * line_height
        cv2.putText(
            image,
            label,
            (label_x1 + padding, baseline_y),
            font,
            scale,
            (8, 14, 20),
            thickness,
            cv2.LINE_AA,
        )


__all__ = ["SnapshotOverlay", "render_snapshot_annotations"]
