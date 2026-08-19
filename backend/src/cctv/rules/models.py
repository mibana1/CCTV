"""Rule definitions, normalized observations, and emitted domain events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """One persisted rule configuration consumed by a registered evaluator."""

    id: str
    name: str
    source_name: str
    rule_type: str
    class_name: str | None
    geometry: dict[str, Any]
    parameters: dict[str, Any]
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TrackSample:
    """Resolution-independent point and box for one tracked detection."""

    track_id: int
    class_name: str
    confidence: float
    timestamp_seconds: float
    point: tuple[float, float]
    box: tuple[float, float, float, float]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuleEvent:
    """A lifecycle or point-in-time event emitted by a rule evaluator."""

    rule_id: str
    track_id: int
    event_type: str
    event_state: str
    occurred_at_seconds: float
    class_name: str
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


class RuleConfigurationError(ValueError):
    """Raised when a stored rule cannot be safely evaluated."""


class UnsupportedRuleTypeError(RuleConfigurationError):
    """Raised when no evaluator is registered for a rule type."""


__all__ = [
    "RuleConfigurationError",
    "RuleDefinition",
    "RuleEvent",
    "TrackSample",
    "UnsupportedRuleTypeError",
]
