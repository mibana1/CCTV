"""Polygon-based intrusion and loitering evaluators."""

from __future__ import annotations

from dataclasses import dataclass

from cctv.rules.evaluators.base import number_parameter
from cctv.rules.geometry import point_in_polygon, polygon_from_geometry
from cctv.rules.models import RuleDefinition, RuleEvent, TrackSample


@dataclass(slots=True)
class _ZoneState:
    inside_since: float | None = None
    last_seen: float = 0
    last_sample: TrackSample | None = None
    inside: bool = False
    emitted: bool = False
    cooldown_until: float = 0


class ZoneRuleEvaluator:
    """Reusable zone lifecycle evaluator configured by subclasses."""

    rule_type = "zone"
    started_event_type = "zone_started"
    ended_event_type = "zone_ended"
    duration_parameter = "dwell_seconds"
    default_duration_seconds = 0.0

    def __init__(self) -> None:
        self._states: dict[tuple[str, int], _ZoneState] = {}

    def validate(self, rule: RuleDefinition) -> None:
        polygon_from_geometry(rule.geometry)
        self._duration(rule)
        self._missing_tolerance(rule)
        self._cooldown(rule)

    def evaluate(
        self,
        rule: RuleDefinition,
        timestamp_seconds: float,
        samples: tuple[TrackSample, ...],
    ) -> tuple[RuleEvent, ...]:
        polygon = polygon_from_geometry(rule.geometry)
        duration = self._duration(rule)
        missing_tolerance = self._missing_tolerance(rule)
        cooldown = self._cooldown(rule)
        observed_track_ids: set[int] = set()
        events: list[RuleEvent] = []

        for sample in samples:
            observed_track_ids.add(sample.track_id)
            key = (rule.id, sample.track_id)
            state = self._states.get(key)
            inside = point_in_polygon(sample.point, polygon)
            if state is None:
                if not inside:
                    continue
                state = _ZoneState()
                self._states[key] = state

            state.last_seen = timestamp_seconds
            state.last_sample = sample
            if inside:
                if not state.inside:
                    state.inside = True
                    state.inside_since = timestamp_seconds
                if (
                    not state.emitted
                    and state.inside_since is not None
                    and timestamp_seconds - state.inside_since >= duration
                    and timestamp_seconds >= state.cooldown_until
                ):
                    state.emitted = True
                    events.append(
                        self._event(
                            rule,
                            sample,
                            self.started_event_type,
                            "started",
                            timestamp_seconds,
                            reason="condition_met",
                            duration_seconds=timestamp_seconds - state.inside_since,
                        )
                    )
                continue

            events.extend(
                self._end_state(
                    rule,
                    state,
                    timestamp_seconds,
                    cooldown,
                    reason="outside",
                )
            )

        for key, state in tuple(self._states.items()):
            if key[0] != rule.id or key[1] in observed_track_ids:
                continue
            if state.inside and timestamp_seconds - state.last_seen > missing_tolerance:
                events.extend(
                    self._end_state(
                        rule,
                        state,
                        timestamp_seconds,
                        cooldown,
                        reason="missing",
                    )
                )
            if (
                not state.inside
                and not state.emitted
                and timestamp_seconds > state.cooldown_until + missing_tolerance
            ):
                del self._states[key]
        return tuple(events)

    def _end_state(
        self,
        rule: RuleDefinition,
        state: _ZoneState,
        timestamp_seconds: float,
        cooldown: float,
        *,
        reason: str,
    ) -> tuple[RuleEvent, ...]:
        events: tuple[RuleEvent, ...] = ()
        if state.emitted and state.last_sample is not None:
            events = (
                self._event(
                    rule,
                    state.last_sample,
                    self.ended_event_type,
                    "ended",
                    timestamp_seconds,
                    reason=reason,
                    duration_seconds=(
                        timestamp_seconds - state.inside_since
                        if state.inside_since is not None
                        else 0
                    ),
                ),
            )
        state.inside = False
        state.inside_since = None
        state.emitted = False
        state.cooldown_until = timestamp_seconds + cooldown
        return events

    def _event(
        self,
        rule: RuleDefinition,
        sample: TrackSample,
        event_type: str,
        event_state: str,
        timestamp_seconds: float,
        **payload: object,
    ) -> RuleEvent:
        return RuleEvent(
            rule_id=rule.id,
            track_id=sample.track_id,
            event_type=event_type,
            event_state=event_state,
            occurred_at_seconds=timestamp_seconds,
            class_name=sample.class_name,
            confidence=sample.confidence,
            payload={
                "point": list(sample.point),
                "box": list(sample.box),
                "rule_type": rule.rule_type,
                **payload,
            },
        )

    def _duration(self, rule: RuleDefinition) -> float:
        return number_parameter(
            rule,
            self.duration_parameter,
            self.default_duration_seconds,
        )

    @staticmethod
    def _missing_tolerance(rule: RuleDefinition) -> float:
        return number_parameter(rule, "missing_tolerance_seconds", 2.0, maximum=300)

    @staticmethod
    def _cooldown(rule: RuleDefinition) -> float:
        return number_parameter(rule, "cooldown_seconds", 5.0, maximum=3_600)


class IntrusionEvaluator(ZoneRuleEvaluator):
    """Emit an intrusion lifecycle when a track enters a configured polygon."""

    rule_type = "intrusion"
    started_event_type = "intrusion_started"
    ended_event_type = "intrusion_ended"
    duration_parameter = "dwell_seconds"
    default_duration_seconds = 0.0


class LoiteringEvaluator(ZoneRuleEvaluator):
    """Emit a loitering lifecycle after continuous polygon dwell time."""

    rule_type = "loitering"
    started_event_type = "loitering_started"
    ended_event_type = "loitering_ended"
    duration_parameter = "duration_seconds"
    default_duration_seconds = 30.0


__all__ = ["IntrusionEvaluator", "LoiteringEvaluator", "ZoneRuleEvaluator"]
