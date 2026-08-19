"""RTSP source validation and authenticated encryption."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

MAX_RTSP_URL_LENGTH = 2_048


class CameraCredentialError(ValueError):
    """Raised when a camera source or encryption key is invalid."""


@dataclass(frozen=True, slots=True)
class PreparedRtspSource:
    """A private source URL and its credential-free display endpoint."""

    source_url: str
    endpoint: str


class CameraCredentialCipher:
    """Encrypt source URLs before they cross the repository boundary."""

    def __init__(self, key: str | bytes) -> None:
        encoded_key = key.encode("ascii") if isinstance(key, str) else key
        try:
            self._fernet = Fernet(encoded_key)
        except (TypeError, ValueError) as error:
            raise CameraCredentialError("camera credential key is invalid") from error

    @classmethod
    def from_settings(
        cls,
        *,
        configured_key: SecretStr | None,
        key_path: Path,
    ) -> CameraCredentialCipher:
        if configured_key is not None and configured_key.get_secret_value():
            return cls(configured_key.get_secret_value())
        return cls(_load_or_create_key(key_path))

    def encrypt(self, source_url: str) -> str:
        return self._fernet.encrypt(source_url.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeError, ValueError) as error:
            raise CameraCredentialError("stored camera credentials cannot be decrypted") from error


def prepare_rtsp_source(
    rtsp_url: str,
    *,
    username: str | None = None,
    password: str | None = None,
) -> PreparedRtspSource:
    """Validate an RTSP endpoint and insert separately supplied credentials."""
    normalized_url = rtsp_url.strip()
    if not normalized_url or len(normalized_url) > MAX_RTSP_URL_LENGTH:
        raise CameraCredentialError(f"RTSP URL must contain 1-{MAX_RTSP_URL_LENGTH} characters")
    try:
        parsed = urlsplit(normalized_url)
        port = parsed.port
    except ValueError as error:
        raise CameraCredentialError("RTSP URL contains an invalid host or port") from error
    if parsed.scheme.casefold() not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise CameraCredentialError("RTSP URL must use rtsp:// or rtsps:// and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise CameraCredentialError("enter the RTSP account in the separate username and password fields")
    if parsed.fragment:
        raise CameraCredentialError("RTSP URL must not include a fragment")

    normalized_username = username.strip() if username is not None else None
    supplied_username = bool(normalized_username)
    supplied_password = password is not None and bool(password)
    if supplied_username != supplied_password:
        raise CameraCredentialError("RTSP username and password must be supplied together")

    hostname = parsed.hostname
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    host_and_port = f"{display_host}:{port}" if port is not None else display_host
    source_netloc = host_and_port
    if supplied_username and password is not None:
        source_netloc = (
            f"{quote(normalized_username or '', safe='')}:{quote(password, safe='')}@{host_and_port}"
        )
    source_url = urlunsplit(
        (parsed.scheme.casefold(), source_netloc, parsed.path, parsed.query, "")
    )
    endpoint = urlunsplit((parsed.scheme.casefold(), host_and_port, parsed.path, "", ""))
    if len(source_url) > MAX_RTSP_URL_LENGTH:
        raise CameraCredentialError(
            f"RTSP URL with credentials must not exceed {MAX_RTSP_URL_LENGTH} characters"
        )
    return PreparedRtspSource(source_url=source_url, endpoint=endpoint)


def _load_or_create_key(key_path: Path) -> bytes:
    path = Path(key_path).expanduser().resolve()
    try:
        return path.read_bytes().strip()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)

    generated = Fernet.generate_key()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes().strip()
    try:
        os.write(descriptor, generated + b"\n")
    finally:
        os.close(descriptor)
    return generated


__all__ = [
    "MAX_RTSP_URL_LENGTH",
    "CameraCredentialCipher",
    "CameraCredentialError",
    "PreparedRtspSource",
    "prepare_rtsp_source",
]
