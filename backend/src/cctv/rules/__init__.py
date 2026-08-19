"""Extensible track-based event rule engine."""

from cctv.rules.engine import RuleEngine
from cctv.rules.evaluators import (
    IntrusionEvaluator,
    LineCrossingEvaluator,
    LoiteringEvaluator,
    RuleEvaluator,
    VisualColorEvaluator,
    built_in_evaluators,
)
from cctv.rules.models import (
    RuleConfigurationError,
    RuleDefinition,
    RuleEvent,
    TrackSample,
    UnsupportedRuleTypeError,
)

__all__ = [
    "IntrusionEvaluator",
    "LineCrossingEvaluator",
    "LoiteringEvaluator",
    "RuleConfigurationError",
    "RuleDefinition",
    "RuleEngine",
    "RuleEvaluator",
    "RuleEvent",
    "TrackSample",
    "UnsupportedRuleTypeError",
    "VisualColorEvaluator",
    "built_in_evaluators",
]
