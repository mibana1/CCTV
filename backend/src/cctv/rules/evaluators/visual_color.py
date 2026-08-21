"""Track-window evaluation for user-defined upper-body colors."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from cctv.rules.evaluators.base import integer_parameter, number_parameter
from cctv.rules.models import RuleConfigurationError, RuleDefinition, RuleEvent, TrackSample
from cctv.vision import SUPPORTED_UPPER_BODY_COLORS

_COLOR_ALIASES = {
    "빨강": "red",
    "빨간색": "red",
    "주황": "orange",
    "주황색": "orange",
    "노랑": "yellow",
    "노란색": "yellow",
    "초록": "green",
    "초록색": "green",
    "파랑": "blue",
    "파란색": "blue",
    "보라": "purple",
    "보라색": "purple",
    "분홍": "pink",
    "분홍색": "pink",
    "검정": "black",
    "검은색": "black",
    "흰색": "white",
    "하양": "white",
    "회색": "gray",
}


@dataclass(frozen=True, slots=True)
class _ColorVote:
    matches: bool
    color: str
    confidence: float
    crop_box: list[int]
    distribution: dict[str, float]


@dataclass(slots=True)
class _ColorState:
    votes: deque[_ColorVote]
    last_seen: float
    last_sample: TrackSample
    active: bool = False
    cooldown_until: float = 0


class VisualColorEvaluator:
    """Emit one point event when a target torso color wins an M-of-N track window."""

    rule_type = "visual_color"

    def __init__(self) -> None:
        self._states: dict[tuple[str, int], _ColorState] = {}

    def validate(self, rule: RuleDefinition) -> None:
        if rule.class_name is not None and rule.class_name.casefold() != "person":
            raise RuleConfigurationError("visual_color class_name must be person")
        if rule.geometry:
            raise RuleConfigurationError("visual_color geometry must be empty")
        self._target_color(rule)
        window_size = self._window_size(rule)
        minimum_matches = self._minimum_matches(rule)
        if minimum_matches > window_size:
            raise RuleConfigurationError("minimum_matches must not exceed window_size")
        self._minimum_confidence(rule)
        self._missing_tolerance(rule)
        self._cooldown(rule)

    def evaluate(
        self,
        rule: RuleDefinition,
        timestamp_seconds: float,
        samples: tuple[TrackSample, ...],
    ) -> tuple[RuleEvent, ...]:
        target_color = self._target_color(rule)
        window_size = self._window_size(rule)
        minimum_matches = self._minimum_matches(rule)
        minimum_confidence = self._minimum_confidence(rule)
        missing_tolerance = self._missing_tolerance(rule)
        cooldown = self._cooldown(rule)
        observed_track_ids: set[int] = set()
        events: list[RuleEvent] = []

        for sample in samples:
            observed_track_ids.add(sample.track_id)
            key = (rule.id, sample.track_id)
            state = self._states.get(key)
            if state is None:
                state = _ColorState(
                    votes=deque(maxlen=window_size),
                    last_seen=timestamp_seconds,
                    last_sample=sample,
                )
                self._states[key] = state
            state.last_seen = timestamp_seconds
            state.last_sample = sample

            vote = _vote_from_sample(sample, target_color, minimum_confidence)
            if vote is None:
                continue
            state.votes.append(vote)
            match_count = sum(item.matches for item in state.votes)
            if (
                not state.active
                and len(state.votes) >= minimum_matches
                and match_count >= minimum_matches
                and timestamp_seconds >= state.cooldown_until
            ):
                state.active = True
                state.cooldown_until = timestamp_seconds + cooldown
                events.append(
                    self._event(
                        rule,
                        state,
                        target_color,
                        timestamp_seconds,
                        reason="vote_threshold_met",
                    )
                )
            elif (
                state.active
                and len(state.votes) == window_size
                and match_count < minimum_matches
            ):
                self._reset_state(state)

        for key, state in tuple(self._states.items()):
            if key[0] != rule.id or key[1] in observed_track_ids:
                continue
            if timestamp_seconds - state.last_seen <= missing_tolerance:
                continue
            if state.active:
                self._reset_state(state)
            if timestamp_seconds >= state.cooldown_until:
                del self._states[key]
        return tuple(events)

    @staticmethod
    def _reset_state(state: _ColorState) -> None:
        state.active = False
        state.votes.clear()

    @staticmethod
    def _event(
        rule: RuleDefinition,
        state: _ColorState,
        target_color: str,
        timestamp_seconds: float,
        *,
        reason: str,
    ) -> RuleEvent:
        matching = [vote for vote in state.votes if vote.matches]
        latest = state.votes[-1] if state.votes else None
        confidence = (
            sum(vote.confidence for vote in matching) / len(matching)
            if matching
            else latest.confidence if latest is not None else state.last_sample.confidence
        )
        return RuleEvent(
            rule_id=rule.id,
            track_id=state.last_sample.track_id,
            event_type="upper_body_color_detected",
            event_state="occurred",
            occurred_at_seconds=timestamp_seconds,
            class_name=state.last_sample.class_name,
            confidence=round(float(confidence), 6),
            payload={
                "rule_type": rule.rule_type,
                "target": "upper_body",
                "target_color": target_color,
                "observed_color": latest.color if latest is not None else None,
                "matching_votes": len(matching),
                "observed_votes": len(state.votes),
                "window_size": state.votes.maxlen,
                "minimum_matches": VisualColorEvaluator._minimum_matches(rule),
                "minimum_color_confidence": VisualColorEvaluator._minimum_confidence(rule),
                "crop_box": latest.crop_box if latest is not None else None,
                "color_distribution": latest.distribution if latest is not None else {},
                "reason": reason,
                "box": list(state.last_sample.box),
            },
        )

    @staticmethod
    def _target_color(rule: RuleDefinition) -> str:
        value = rule.parameters.get("target_color")
        if not isinstance(value, str) or not value.strip():
            raise RuleConfigurationError("target_color must be a supported color")
        normalized = value.strip().casefold()
        normalized = _COLOR_ALIASES.get(normalized, normalized)
        if normalized not in SUPPORTED_UPPER_BODY_COLORS:
            supported = ", ".join(SUPPORTED_UPPER_BODY_COLORS)
            raise RuleConfigurationError(f"target_color must be one of: {supported}")
        return normalized

    @staticmethod
    def _window_size(rule: RuleDefinition) -> int:
        return integer_parameter(rule, "window_size", 5, minimum=1, maximum=30)

    @staticmethod
    def _minimum_matches(rule: RuleDefinition) -> int:
        return integer_parameter(rule, "minimum_matches", 3, minimum=1, maximum=30)

    @staticmethod
    def _minimum_confidence(rule: RuleDefinition) -> float:
        return number_parameter(rule, "minimum_color_confidence", 0.35, maximum=1)

    @staticmethod
    def _missing_tolerance(rule: RuleDefinition) -> float:
        return number_parameter(rule, "missing_tolerance_seconds", 2.0, maximum=300)

    @staticmethod
    def _cooldown(rule: RuleDefinition) -> float:
        return number_parameter(rule, "cooldown_seconds", 30.0, maximum=3_600)


def _vote_from_sample(
    sample: TrackSample,
    target_color: str,
    minimum_confidence: float,
) -> _ColorVote | None:
    color = sample.attributes.get("upper_body_color")
    confidence = sample.attributes.get("upper_body_color_confidence")
    if not isinstance(color, str):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    normalized_confidence = float(confidence)
    if not 0 <= normalized_confidence <= 1 or normalized_confidence < minimum_confidence:
        return None
    raw_crop_box = sample.attributes.get("upper_body_crop_box")
    crop_box = (
        [int(value) for value in raw_crop_box]
        if isinstance(raw_crop_box, list)
        and len(raw_crop_box) == 4
        and all(isinstance(value, int) for value in raw_crop_box)
        else []
    )
    raw_distribution = sample.attributes.get("upper_body_color_distribution")
    distribution = (
        {
            str(key): float(value)
            for key, value in raw_distribution.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if isinstance(raw_distribution, dict)
        else {}
    )
    normalized_color = color.strip().casefold()
    return _ColorVote(
        matches=normalized_color == target_color,
        color=normalized_color,
        confidence=normalized_confidence,
        crop_box=crop_box,
        distribution=distribution,
    )


__all__ = ["VisualColorEvaluator"]
