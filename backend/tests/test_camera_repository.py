import sqlite3
from pathlib import Path

import pytest

from cctv.db import CameraProvisioningStatus, CameraRepository, initialize_database


def _repository(tmp_path: Path) -> CameraRepository:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    return CameraRepository(database_path)


def test_camera_repository_crud_and_filters(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    first = repository.create_camera(
        name="Entrance",
        location="Lobby",
        stream_path="camera",
    )
    second = repository.create_camera(
        name="Parking",
        stream_path="building/parking",
        enabled=False,
    )

    assert repository.get_camera(first.id) == first
    assert first.source_on_demand is False
    assert repository.list_cameras().total == 2
    assert repository.list_cameras(enabled=True).items == (first,)
    assert repository.list_cameras(enabled=False).items == (second,)

    updated = repository.update_camera(
        first.id,
        {"name": "Main entrance", "location": None, "enabled": False},
    )

    assert updated.name == "Main entrance"
    assert updated.location is None
    assert updated.enabled is False
    assert repository.delete_camera(second.id) is True
    assert repository.delete_camera(second.id) is False


@pytest.mark.parametrize(
    "stream_path",
    ("", "/", "../camera", "camera/../secret", "camera path", "camera?token=secret"),
)
def test_camera_repository_rejects_unsafe_stream_paths(
    tmp_path: Path,
    stream_path: str,
) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(ValueError):
        repository.create_camera(name="Unsafe", stream_path=stream_path)


def test_camera_repository_rejects_duplicate_stream_paths_case_insensitively(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    repository.create_camera(name="First", stream_path="Camera")

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        repository.create_camera(name="Second", stream_path="camera")


def test_camera_repository_keeps_encrypted_source_internal(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    camera = repository.create_camera(
        name="Managed camera",
        stream_path="cam-123456789abc",
        rtsp_endpoint="rtsp://camera.local:554/stream",
        rtsp_source_ciphertext="encrypted-source",
        source_on_demand=True,
        provisioning_status=CameraProvisioningStatus.PENDING,
    )

    source = repository.get_camera_source(camera.id)
    synced = repository.set_provisioning_result(
        camera.id,
        provisioning_status=CameraProvisioningStatus.ACTIVE,
    )

    assert camera.credentials_configured is True
    assert camera.rtsp_endpoint == "rtsp://camera.local:554/stream"
    assert not hasattr(camera, "rtsp_source_ciphertext")
    assert source is not None
    assert source.ciphertext == "encrypted-source"
    assert synced.provisioning_status is CameraProvisioningStatus.ACTIVE
    assert synced.last_synced_at is not None
