"""Fail-safe execution policy for external side effects."""

import logging

from cctv.core.settings import Settings

logger = logging.getLogger(__name__)


def external_action_allowed(settings: Settings, *, action: str) -> bool:
    """Allow external work only in LIVE mode and audit DRY RUN skips."""
    if settings.external_actions_enabled:
        return True

    logger.info(
        "External action skipped",
        extra={
            "event": "external_action_skipped",
            "action": action,
            "reason": "dry_run",
            "app_mode": settings.app_mode,
        },
    )
    return False
