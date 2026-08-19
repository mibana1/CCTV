from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from cctv.cameras import (
    CameraCredentialCipher,
    CameraCredentialError,
    prepare_rtsp_source,
)


def test_prepare_rtsp_source_encodes_credentials_and_redacts_endpoint() -> None:
    prepared = prepare_rtsp_source(
        "rtsp://camera.local:554/live/main?profile=high",
        username="operator@example.com",
        password="p:a/ss#word",
    )

    assert prepared.source_url == (
        "rtsp://operator%40example.com:p%3Aa%2Fss%23word@camera.local:554/"
        "live/main?profile=high"
    )
    assert prepared.endpoint == "rtsp://camera.local:554/live/main"


@pytest.mark.parametrize(
    ("url", "username", "password"),
    [
        ("http://camera.local/stream", None, None),
        ("rtsp://user:pass@camera.local/stream", None, None),
        ("rtsp://camera.local/stream", "user", None),
        ("rtsp://camera.local/stream", None, "password"),
    ],
)
def test_prepare_rtsp_source_rejects_invalid_or_ambiguous_credentials(
    url: str,
    username: str | None,
    password: str | None,
) -> None:
    with pytest.raises(CameraCredentialError):
        prepare_rtsp_source(url, username=username, password=password)


def test_camera_cipher_round_trips_without_plaintext_storage() -> None:
    cipher = CameraCredentialCipher(Fernet.generate_key())
    source = "rtsp://user:password@camera.local:554/stream"

    ciphertext = cipher.encrypt(source)

    assert source not in ciphertext
    assert cipher.decrypt(ciphertext) == source


def test_camera_cipher_generates_and_reuses_private_key_file(tmp_path: Path) -> None:
    key_path = tmp_path / "secrets" / "camera.key"

    first = CameraCredentialCipher.from_settings(configured_key=None, key_path=key_path)
    ciphertext = first.encrypt("rtsp://camera.local/stream")
    second = CameraCredentialCipher.from_settings(configured_key=None, key_path=key_path)

    assert key_path.is_file()
    assert second.decrypt(ciphertext) == "rtsp://camera.local/stream"

