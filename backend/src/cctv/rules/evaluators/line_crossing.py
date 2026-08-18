"""Directed line-crossing evaluator with hysteresis and cooldown."""

from __future__ import annotations

from dataclasses import dataclass

from cctv.rules.evaluators.base import choice_parameter, number_parameter
from cctv.rules.geometry import line_from_geometry, segments_intersect, signed_line_distance
from cctv.rules.models import RuleDefinition, RuleEvent, TrackSample

_DIRECTIONS = {"any", "negative_to_positive", "positive_to_negative"}


@dataclass(slots=True)
class _LineState:
    point: tuple[float, float]
    side: int
    last_seen: float
    cooldown_until: float = 0


class LineCrossingEvaluator:
    """Emit one point-in-time event when a track crosses a finite line segment."""

    rule_type = "line_crossing"

    def __init__(self) -> None:
        self._states: dict[tuple[str, int], _LineState] = {}

    def validate(self, rule: RuleDefinition) -> None:
        line_from_geometry(rule.geometry)
        self._hysteresis(rule)
        self._cooldown(rule)
        self._missing_tolerance(rule)
        self._direction(rule)

    def evaluate(
        self,
        rule: RuleDefinition,
        timestamp_seconds: float,
        samples: tuple[TrackSample, ...],
    ) -> tuple[RuleEvent, ...]:
        line_start, line_end = line_from_geometry(rule.geometry)
        hysteresis = self._hysteresis(rule)
        cooldown = self._cooldown(rule)
        missing_tolerance = self._missing_tolerance(rule)
        allowed_direction = self._direction(rule)
        observed_track_ids: set[int] = set()
        events: list[RuleEvent] = []

        for sample in samples:
            observed_track_ids.add(sample.track_id)
            distance = signed_line_distance(sample.point, line_start, line_end)
            side = 1 if distance > hysteresis else -1 if distance < -hysteresis else 0
            key = (rule.id, sample.track_id)
            state = self._states.get(key)
            if state is None:
                if side != 0:
                    self._states[key] = _LineState(
                        point=sample.point,
                        side=side,
                        last_seen=timestamp_seconds,
                    )
                continue
            state.last_seen = timestamp_seconds
            if side == 0:
                continue

            direction = "negative_to_positive" if state.side < side else "positive_to_negative"
            crossed = state.side != side and segments_intersect(
                state.point,
                sample.point,
                line_start,
                line_end,
            )
            if (
                crossed
                and timestamp_seconds >= state.cooldown_until
                and allowed_direction in {"any", direction}
            ):
                events.append(
                    RuleEvent(
                        rule_id=rule.id,
                        track_id=sample.track_id,
                        event_type="line_crossed",
                        event_state="occurred",
                        occurred_at_seconds=timestamp_seconds,
                        class_name=sample.class_name,
                        confidence=sample.confidence,
                        payload={
                            "rule_type": rule.rule_type,
                            "direction": direction,
                            "previous_point": list(state.point),
                            "point": list(sample.point),
                            "box": list(sample.box),
                        },
                    )
                )
                state.cooldown_until = timestamp_seconds + cooldown
            state.point = sample.point
            state.side = side

        for key, state in tuple(self._states.items()):
            if key[0] != rule.id or key[1] in observed_track_ids:
                continue
            if timestamp_seconds - state.last_seen > missing_tolerance:
                del self._states[key]
        return tuple(events)

    @staticmethod
    def _hysteresis(rule: RuleDefinition) -> float:
        return number_parameter(rule, "hysteresis", 0.005, maximum=0.25)

    @staticmethod
    def _cooldown(rule: RuleDefinition) -> float:
        return number_parameter(rule, "cooldown_seconds", 2.0, maximum=3_600)

    @staticmethod
    def _missing_tolerance(rule: RuleDefinition) -> float:
        return number_parameter(rule, "missing_tolerance_seconds", 2.0, maximum=300)

    @staticmethod
    def _direction(rule: RuleDefinition) -> str:
        return choice_parameter(rule, "direction", "any", _DIRECTIONS)


__all__ = ["LineCrossingEvaluator"]
