"""Evaluator contract and shared configuration validation."""

from __future__ import annotations

from math import isfinite
from typing import Protocol

from cctv.rules.models import RuleConfigurationError, RuleDefinition, RuleEvent, TrackSample


class RuleEvaluator(Protocol):
    """Extension point implemented by each independent rule type."""

    rule_type: str

    def validate(self, rule: RuleDefinition) -> None:
        """Reject invalid geometry or parameters before stream processing starts."""

    def evaluate(
        self,
        rule: RuleDefinition,
        timestamp_seconds: float,
        samples: tuple[TrackSample, ...],
    ) -> tuple[RuleEvent, ...]:
        """Evaluate one sampled frame and update evaluator-owned runtime state."""


def number_parameter(
    rule: RuleDefinition,
    name: str,
    default: float,
    *,
    minimum: float = 0,
    maximum: float = 86_400,
) -> float:
    value = rule.parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuleConfigurationError(f"{name} must be a number")
    normalized = float(value)
    if not isfinite(normalized) or not minimum <= normalized <= maximum:
        raise RuleConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def choice_parameter(
    rule: RuleDefinition,
    name: str,
    default: str,
    choices: set[str],
) -> str:
    value = rule.parameters.get(name, default)
    if not isinstance(value, str) or value not in choices:
        joined = ", ".join(sorted(choices))
        raise RuleConfigurationError(f"{name} must be one of: {joined}")
    return value


def integer_parameter(
    rule: RuleDefinition,
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: int = 10_000,
) -> int:
    value = rule.parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuleConfigurationError(f"{name} must be a whole number")
    if not minimum <= value <= maximum:
        raise RuleConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


__all__ = ["RuleEvaluator", "choice_parameter", "integer_parameter", "number_parameter"]
