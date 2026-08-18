"""Built-in evaluator registry entries."""

from cctv.rules.evaluators.base import RuleEvaluator
from cctv.rules.evaluators.line_crossing import LineCrossingEvaluator
from cctv.rules.evaluators.zone import IntrusionEvaluator, LoiteringEvaluator


def built_in_evaluators() -> tuple[RuleEvaluator, ...]:
    """Return fresh stateful evaluators for one rule-engine instance."""
    return (IntrusionEvaluator(), LineCrossingEvaluator(), LoiteringEvaluator())


__all__ = [
    "IntrusionEvaluator",
    "LineCrossingEvaluator",
    "LoiteringEvaluator",
    "RuleEvaluator",
    "built_in_evaluators",
]
