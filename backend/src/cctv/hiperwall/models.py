"""Model-neutral Hiperwall display actions derived from rule events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class DisplayAction:
    """One auditable Hiperwall operation persisted before external execution."""

    rule_event_id: str
    action_type: Literal["open_source", "restore_layout"]
    request: dict[str, Any]
    result: dict[str, Any]
    mode: Literal["dry_run", "live"] = "dry_run"
    status: Literal["simulated", "pending"] = "simulated"
    available_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: str = field(default_factory=lambda: str(uuid4()))


__all__ = ["DisplayAction"]
