"""Validated rule-to-Hiperwall content and Zone mappings."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from cctv.rules import RuleDefinition


class HiperwallMappingError(ValueError):
    """Raised when a rule contains an unsafe Hiperwall mapping."""


@dataclass(frozen=True, slots=True)
class HiperwallLayout:
    mode: str
    x: float
    y: float
    width: float
    height: float

    def as_request(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class HiperwallMapping:
    selector: str
    content: str
    zone_id: str | None
    layout: HiperwallLayout | None
    display_seconds: int

    def target_request(self) -> dict[str, Any]:
        target: dict[str, Any] = {
            "selector": self.selector,
            "value": self.content,
            "zone_id": self.zone_id,
        }
        if self.layout is not None:
            target["layout"] = self.layout.as_request()
        return target


def mapping_from_rule(
    rule: RuleDefinition,
    *,
    default_display_seconds: int,
) -> HiperwallMapping | None:
    """Read an optional ``parameters.hiperwall`` mapping from a rule."""
    raw = rule.parameters.get("hiperwall")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise HiperwallMappingError("parameters.hiperwall must be an object")
    if raw.get("enabled", True) is False:
        return None
    if raw.get("enabled", True) is not True:
        raise HiperwallMappingError("parameters.hiperwall.enabled must be a boolean")

    content_name = _optional_text(raw.get("content_name"), "content_name")
    content_uuid = _optional_text(raw.get("content_uuid"), "content_uuid")
    if bool(content_name) == bool(content_uuid):
        raise HiperwallMappingError(
            "parameters.hiperwall must define exactly one of content_name or content_uuid"
        )
    selector = "uuid" if content_uuid else "name"
    content = content_uuid or content_name
    if content is None:  # pragma: no cover - guarded above
        raise HiperwallMappingError("Hiperwall content mapping is missing")

    zone_id = _optional_text(raw.get("zone_id"), "zone_id")
    layout = _layout(raw.get("layout"), zone_id)
    display_seconds = raw.get("display_seconds", default_display_seconds)
    if isinstance(display_seconds, bool) or not isinstance(display_seconds, int):
        raise HiperwallMappingError(
            "parameters.hiperwall.display_seconds must be a whole number"
        )
    if not 1 <= display_seconds <= 86_400:
        raise HiperwallMappingError(
            "parameters.hiperwall.display_seconds must be between 1 and 86400"
        )
    return HiperwallMapping(
        selector=selector,
        content=content,
        zone_id=zone_id,
        layout=layout,
        display_seconds=display_seconds,
    )


def validate_rule_hiperwall_mapping(
    rule: RuleDefinition,
    *,
    default_display_seconds: int = 30,
) -> None:
    mapping_from_rule(rule, default_display_seconds=default_display_seconds)


def _layout(value: object, zone_id: str | None) -> HiperwallLayout | None:
    if value is None:
        return None
    if not zone_id:
        raise HiperwallMappingError("parameters.hiperwall.layout requires zone_id")
    if not isinstance(value, dict):
        raise HiperwallMappingError("parameters.hiperwall.layout must be an object")
    mode = str(value.get("mode", "percent")).strip().lower()
    if mode not in {"percent", "pixels"}:
        raise HiperwallMappingError("parameters.hiperwall.layout.mode must be percent or pixels")
    x = _number(value.get("x"), "x")
    y = _number(value.get("y"), "y")
    width = _number(value.get("width"), "width")
    height = _number(value.get("height"), "height")
    if width <= 0 or height <= 0:
        raise HiperwallMappingError("Hiperwall layout width and height must be greater than zero")
    if mode == "percent" and (x < 0 or y < 0 or x + width > 100 or y + height > 100):
        raise HiperwallMappingError(
            "percent Hiperwall layout must remain inside the selected Zone"
        )
    return HiperwallLayout(mode=mode, x=x, y=y, width=width, height=height)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise HiperwallMappingError(f"parameters.hiperwall.layout.{field} must be a number")
    try:
        normalized = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise HiperwallMappingError(
            f"parameters.hiperwall.layout.{field} must be a number"
        ) from error
    if not isfinite(normalized):
        raise HiperwallMappingError(
            f"parameters.hiperwall.layout.{field} must be finite"
        )
    return normalized


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HiperwallMappingError(f"parameters.hiperwall.{field} must be text")
    normalized = value.strip()
    if not normalized:
        raise HiperwallMappingError(f"parameters.hiperwall.{field} must not be empty")
    if len(normalized) > 512:
        raise HiperwallMappingError(f"parameters.hiperwall.{field} is too long")
    return normalized


__all__ = [
    "HiperwallLayout",
    "HiperwallMapping",
    "HiperwallMappingError",
    "mapping_from_rule",
    "validate_rule_hiperwall_mapping",
]
