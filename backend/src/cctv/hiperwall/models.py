"""Model-neutral Hiperwall display actions derived from rule events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class DisplayAction:
    """One auditable Hiperwall operation produced without sending a request."""

    rule_event_id: str
    action_type: Literal["open_source", "restore_layout"]
    request: dict[str, Any]
    result: dict[str, Any]
    mode: Literal["dry_run"] = "dry_run"
    status: Literal["simulated"] = "simulated"
    id: str = field(default_factory=lambda: str(uuid4()))


__all__ = ["DisplayAction"]
