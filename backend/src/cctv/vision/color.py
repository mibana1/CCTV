"""OpenCV-only upper-body color classification for tracked people."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cctv.inference import BoundingBox, FrameDetections
from cctv.media import DecodedFrame

SUPPORTED_UPPER_BODY_COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
)


@dataclass(frozen=True, slots=True)
class UpperBodyColorObservation:
    """One dominant-color result from the torso portion of a person track."""

    track_id: int
    color: str
    confidence: float
    crop_box: BoundingBox
    pixel_count: int
    distribution: dict[str, float]

    @property
    def rule_attributes(self) -> dict[str, object]:
        """Return JSON-compatible attributes consumed by visual rules."""
        return {
            "upper_body_color": self.color,
            "upper_body_color_confidence": self.confidence,
            "upper_body_color_distribution": self.distribution,
            "upper_body_color_pixel_count": self.pixel_count,
            "upper_body_crop_box": [
                self.crop_box.x1,
                self.crop_box.y1,
                self.crop_box.x2,
                self.crop_box.y2,
            ],
        }


@dataclass(frozen=True, slots=True)
class UpperBodyColorSummary:
    processed_frames: int
    person_detections: int
    classified_crops: int
    skipped_crops: int
    color_counts: dict[str, int]


class UpperBodyColorAnalyzer:
    """Classify dominant torso colors without a generative or learned model."""

    def __init__(
        self,
        *,
        minimum_crop_width: int = 8,
        minimum_crop_height: int = 8,
    ) -> None:
        if minimum_crop_width < 1 or minimum_crop_height < 1:
            raise ValueError("minimum crop dimensions must be positive")
        self.minimum_crop_width = minimum_crop_width
        self.minimum_crop_height = minimum_crop_height
        self._processed_frames = 0
        self._person_detections = 0
        self._classified_crops = 0
        self._skipped_crops = 0
        self._color_counts: Counter[str] = Counter()

    @property
    def summary(self) -> UpperBodyColorSummary:
        return UpperBodyColorSummary(
            processed_frames=self._processed_frames,
            person_detections=self._person_detections,
            classified_crops=self._classified_crops,
            skipped_crops=self._skipped_crops,
            color_counts=dict(sorted(self._color_counts.items())),
        )

    def analyze(
        self,
        frame: DecodedFrame,
        result: FrameDetections,
    ) -> dict[int, UpperBodyColorObservation]:
        """Return at most one torso-color observation per tracked person."""
        if frame.source_index != result.source_index or frame.sample_index != result.sample_index:
            raise ValueError("color analysis frame and detections must refer to the same sample")
        if frame.image.ndim != 3 or frame.image.shape[2] != 3:
            raise ValueError("upper-body color analysis requires a BGR image")

        self._processed_frames += 1
        observations: dict[int, UpperBodyColorObservation] = {}
        for detection in result.detections:
            if detection.label.strip().casefold() != "person" or detection.track_id is None:
                continue
            self._person_detections += 1
            crop_box = _upper_body_box(
                detection.box,
                frame_width=frame.image.shape[1],
                frame_height=frame.image.shape[0],
            )
            width = crop_box.x2 - crop_box.x1
            height = crop_box.y2 - crop_box.y1
            if width < self.minimum_crop_width or height < self.minimum_crop_height:
                self._skipped_crops += 1
                continue
            crop = frame.image[crop_box.y1 : crop_box.y2, crop_box.x1 : crop_box.x2]
            classified = _classify_crop(crop)
            if classified is None:
                self._skipped_crops += 1
                continue
            color, confidence, distribution, pixel_count = classified
            observation = UpperBodyColorObservation(
                track_id=detection.track_id,
                color=color,
                confidence=confidence,
                crop_box=crop_box,
                pixel_count=pixel_count,
                distribution=distribution,
            )
            observations[detection.track_id] = observation
            self._classified_crops += 1
            self._color_counts[color] += 1
        return observations


def _upper_body_box(
    person_box: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    """Approximate shoulders-to-waist while avoiding box edges and the head."""
    width = max(0, person_box.x2 - person_box.x1)
    height = max(0, person_box.y2 - person_box.y1)
    x1 = round(person_box.x1 + width * 0.15)
    x2 = round(person_box.x2 - width * 0.15)
    y1 = round(person_box.y1 + height * 0.18)
    y2 = round(person_box.y1 + height * 0.62)
    return BoundingBox(
        x1=max(0, min(frame_width, x1)),
        y1=max(0, min(frame_height, y1)),
        x2=max(0, min(frame_width, x2)),
        y2=max(0, min(frame_height, y2)),
    )


def _classify_crop(
    crop: NDArray[np.uint8],
) -> tuple[str, float, dict[str, float], int] | None:
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    y_grid, x_grid = np.ogrid[:height, :width]
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    radius_x = max(width * 0.5, 1)
    radius_y = max(height * 0.5, 1)
    central_mask = (
        ((x_grid - center_x) / radius_x) ** 2
        + ((y_grid - center_y) / radius_y) ** 2
        <= 1
    )
    hue = hsv[:, :, 0][central_mask]
    saturation = hsv[:, :, 1][central_mask]
    value = hsv[:, :, 2][central_mask]
    pixel_count = int(hue.size)
    if pixel_count == 0:
        return None

    black = value <= 50
    white = (~black) & (saturation <= 40) & (value >= 180)
    gray = (~black) & (~white) & (saturation <= 45)
    chromatic = ~(black | white | gray)
    masks = {
        "red": chromatic & ((hue < 10) | (hue >= 170)),
        "orange": chromatic & (hue >= 10) & (hue < 23),
        "yellow": chromatic & (hue >= 23) & (hue < 35),
        "green": chromatic & (hue >= 35) & (hue < 85),
        "blue": chromatic & (hue >= 85) & (hue < 130),
        "purple": chromatic & (hue >= 130) & (hue < 160),
        "pink": chromatic & (hue >= 160) & (hue < 170),
        "black": black,
        "white": white,
        "gray": gray,
    }
    counts = {color: int(np.count_nonzero(mask)) for color, mask in masks.items()}
    color = max(SUPPORTED_UPPER_BODY_COLORS, key=lambda item: counts[item])
    confidence = counts[color] / pixel_count
    distribution = {
        item: round(counts[item] / pixel_count, 4)
        for item in SUPPORTED_UPPER_BODY_COLORS
        if counts[item]
    }
    return color, round(float(confidence), 6), distribution, pixel_count


__all__ = [
    "SUPPORTED_UPPER_BODY_COLORS",
    "UpperBodyColorAnalyzer",
    "UpperBodyColorObservation",
    "UpperBodyColorSummary",
]
