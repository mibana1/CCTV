"""Normalized 2D geometry helpers shared by independent rule evaluators."""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any

from cctv.rules.models import RuleConfigurationError

Point = tuple[float, float]


def polygon_from_geometry(geometry: dict[str, Any]) -> tuple[Point, ...]:
    """Validate and return a normalized polygon with at least three vertices."""
    if geometry.get("type") != "polygon":
        raise RuleConfigurationError("zone rules require geometry.type=polygon")
    raw_points = geometry.get("points")
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise RuleConfigurationError("polygon geometry requires at least three points")
    return tuple(_normalized_point(point, "polygon point") for point in raw_points)


def line_from_geometry(geometry: dict[str, Any]) -> tuple[Point, Point]:
    """Validate and return two different normalized line endpoints."""
    if geometry.get("type") != "line":
        raise RuleConfigurationError("line crossing rules require geometry.type=line")
    raw_points = geometry.get("points")
    if not isinstance(raw_points, list) or len(raw_points) != 2:
        raise RuleConfigurationError("line geometry requires exactly two points")
    start, end = (_normalized_point(point, "line point") for point in raw_points)
    if start == end:
        raise RuleConfigurationError("line endpoints must be different")
    return start, end


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        px, py = previous
        cx, cy = current
        if (cy > point[1]) != (py > point[1]):
            intersection_x = (px - cx) * (point[1] - cy) / (py - cy) + cx
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def signed_line_distance(point: Point, start: Point, end: Point) -> float:
    """Return signed perpendicular distance to the infinite directed line."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return (dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / hypot(dx, dy)


def segments_intersect(
    first_start: Point, first_end: Point, second_start: Point, second_end: Point
) -> bool:
    """Return whether two closed line segments intersect."""
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if orientations[0] * orientations[1] < 0 and orientations[2] * orientations[3] < 0:
        return True
    return any(
        abs(orientation) <= 1e-9 and _point_on_segment(point, start, end)
        for orientation, point, start, end in (
            (orientations[0], second_start, first_start, first_end),
            (orientations[1], second_end, first_start, first_end),
            (orientations[2], first_start, second_start, second_end),
            (orientations[3], first_end, second_start, second_end),
        )
    )


def _normalized_point(value: object, label: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuleConfigurationError(f"{label} must contain x and y")
    x, y = value
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
    ):
        raise RuleConfigurationError(f"{label} coordinates must be numbers")
    point = (float(x), float(y))
    if not all(isfinite(coordinate) and 0 <= coordinate <= 1 for coordinate in point):
        raise RuleConfigurationError(f"{label} coordinates must be between 0 and 1")
    return point


def _orientation(start: Point, end: Point, point: Point) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if abs(_orientation(start, end, point)) > 1e-9:
        return False
    return (
        min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
        and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
    )


__all__ = [
    "Point",
    "line_from_geometry",
    "point_in_polygon",
    "polygon_from_geometry",
    "segments_intersect",
    "signed_line_distance",
]
