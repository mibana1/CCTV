"""Structured JSON logging with credential redaction and file rotation."""

import json
import logging
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pydantic import SecretStr

LOGGER_NAME = "cctv"
REDACTED = "***"

_STANDARD_RECORD_ATTRIBUTES = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
}
_RESERVED_OUTPUT_FIELDS = {
    "timestamp",
    "level",
    "service",
    "logger",
    "message",
    "exception",
}
_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "passphrase",
    "passwd",
    "password",
    "proxy_authorization",
    "secret",
    "set_cookie",
    "token",
    "username",
    "user_name",
}

_RTSP_CREDENTIALS = re.compile(r"(?i)\b(rtsps?://)([^/@\s]+)@")
_AUTH_SCHEME = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:token|password|passwd|api[_-]?key|authorization|username)=)[^&#\s]+"
)
_INLINE_SECRET = re.compile(
    r"(?i)\b(password|passwd|passphrase|token|api[_-]?key|secret|authorization|username)"
    r"\b(\s*[=:]\s*)([^\s,;]+)"
)
_XML_SECRET = re.compile(
    r"(?is)<(token|password|passwd|passphrase|secret|apiKey|authorization|username)"
    r"\b[^>]*>.*?</\1\s*>"
)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)
    parts = set(normalized.split("_"))
    return (
        normalized in _SENSITIVE_KEY_PARTS
        or bool(
            parts
            & {
                "authorization",
                "cookie",
                "password",
                "passwd",
                "passphrase",
                "secret",
                "token",
                "username",
            }
        )
        or normalized.endswith(("_api_key", "_apikey"))
    )


def redact_text(value: str) -> str:
    """Remove known credential forms from free-form log text."""

    def redact_xml(match: re.Match[str]) -> str:
        tag = match.group(1)
        return f"<{tag}>{REDACTED}</{tag}>"

    redacted = _XML_SECRET.sub(redact_xml, value)
    redacted = _RTSP_CREDENTIALS.sub(rf"\1{REDACTED}:{REDACTED}@", redacted)
    redacted = _AUTH_SCHEME.sub(lambda match: f"{match.group(1)} {REDACTED}", redacted)
    redacted = _QUERY_SECRET.sub(lambda match: f"{match.group(1)}{REDACTED}", redacted)
    return _INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}",
        redacted,
    )


def sanitize_for_logging(value: Any, *, key: str | None = None) -> Any:
    """Recursively convert context values to JSON-safe, redacted values."""
    if key is not None and _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_for_logging(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


class JsonFormatter(logging.Formatter):
    """Render one JSON object per log record."""

    def __init__(self, *, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat(
            timespec="milliseconds"
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp.replace("+00:00", "Z"),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRIBUTES or key in _RESERVED_OUTPUT_FIELDS:
                continue
            payload[key] = sanitize_for_logging(value, key=key)

        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    *,
    level: str,
    log_path: Path,
    max_bytes: int,
    backup_count: int,
    service: str = "cctv-backend",
) -> logging.Logger:
    """Configure idempotent stdout and rotating JSONL handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    shutdown_logging()

    logger.setLevel(level.upper())
    logger.propagate = False
    formatter = JsonFormatter(service=service)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def shutdown_logging() -> None:
    """Flush, close, and remove handlers owned by the CCTV logger."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
