"""Evaluator registry and frame-to-event orchestration."""

from __future__ import annotations

from collections.abc import Iterable

from cctv.inference import FrameDetections
from cctv.rules.evaluators import RuleEvaluator, built_in_evaluators
from cctv.rules.models import (
    RuleDefinition,
    RuleEvent,
    TrackSample,
    UnsupportedRuleTypeError,
)


class RuleEngine:
    """Route normalized tracked samples to independently registered evaluators."""

    def __init__(
        self,
        rules: Iterable[RuleDefinition],
        *,
        evaluators: Iterable[RuleEvaluator] | None = None,
    ) -> None:
        self._evaluators: dict[str, RuleEvaluator] = {}
        for evaluator in evaluators if evaluators is not None else built_in_evaluators():
            self.register_evaluator(evaluator)
        self.rules = tuple(rule for rule in rules if rule.enabled)
        for rule in self.rules:
            self._evaluator(rule).validate(rule)

    @property
    def supported_rule_types(self) -> tuple[str, ...]:
        """Return registered types for validation, API discovery, and diagnostics."""
        return tuple(sorted(self._evaluators))

    def register_evaluator(self, evaluator: RuleEvaluator) -> None:
        """Register one extension without modifying the engine dispatch loop."""
        rule_type = evaluator.rule_type.strip()
        if not rule_type:
            raise ValueError("evaluator rule_type must not be empty")
        if rule_type in self._evaluators:
            raise ValueError(f"duplicate evaluator rule_type: {rule_type}")
        self._evaluators[rule_type] = evaluator

    def process(self, result: FrameDetections) -> tuple[RuleEvent, ...]:
        """Evaluate one chronologically ordered tracked frame against every rule."""
        samples = _track_samples(result)
        events: list[RuleEvent] = []
        for rule in self.rules:
            matching_samples = (
                samples
                if rule.class_name is None
                else tuple(
                    sample
                    for sample in samples
                    if sample.class_name.casefold() == rule.class_name.casefold()
                )
            )
            events.extend(
                self._evaluator(rule).evaluate(
                    rule,
                    result.timestamp_seconds,
                    matching_samples,
                )
            )
        return tuple(events)

    def _evaluator(self, rule: RuleDefinition) -> RuleEvaluator:
        try:
            return self._evaluators[rule.rule_type]
        except KeyError as error:
            supported = ", ".join(self.supported_rule_types)
            raise UnsupportedRuleTypeError(
                f"unsupported rule_type {rule.rule_type!r}; registered types: {supported}"
            ) from error


def _track_samples(result: FrameDetections) -> tuple[TrackSample, ...]:
    if result.frame_width <= 0 or result.frame_height <= 0:
        raise ValueError("rule evaluation requires positive frame dimensions")
    samples: list[TrackSample] = []
    for detection in result.detections:
        if detection.track_id is None:
            continue
        samples.append(
            TrackSample(
                track_id=detection.track_id,
                class_name=detection.label,
                confidence=detection.confidence,
                timestamp_seconds=result.timestamp_seconds,
                point=(
                    ((detection.box.x1 + detection.box.x2) / 2) / result.frame_width,
                    detection.box.y2 / result.frame_height,
                ),
                box=(
                    detection.box.x1 / result.frame_width,
                    detection.box.y1 / result.frame_height,
                    detection.box.x2 / result.frame_width,
                    detection.box.y2 / result.frame_height,
                ),
            )
        )
    return tuple(samples)


__all__ = ["RuleEngine"]
