from pathlib import Path

from cryptography.fernet import Fernet

from cctv.cameras import CameraCredentialCipher, CameraManagementService, MediaMtxPathStatus
from cctv.db import CameraProvisioningStatus, CameraRepository, initialize_database


class FakeMediaMtx:
    def __init__(self) -> None:
        self.paths: dict[str, tuple[str, bool]] = {}
        self.deleted: list[str] = []

    def upsert_rtsp_path(
        self,
        *,
        stream_path: str,
        source_url: str,
        source_on_demand: bool,
    ) -> None:
        self.paths[stream_path] = (source_url, source_on_demand)

    def delete_path(self, stream_path: str) -> None:
        self.deleted.append(stream_path)
        self.paths.pop(stream_path, None)

    def get_path_status(self, stream_path: str) -> MediaMtxPathStatus | None:
        if stream_path not in self.paths:
            return None
        return MediaMtxPathStatus(online=False, available=False)

    def close(self) -> None:
        pass


def _service(tmp_path: Path) -> tuple[CameraManagementService, CameraRepository, FakeMediaMtx]:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = CameraRepository(database_path)
    mediamtx = FakeMediaMtx()
    service = CameraManagementService(
        repository,
        CameraCredentialCipher(Fernet.generate_key()),
        mediamtx,  # type: ignore[arg-type]
    )
    return service, repository, mediamtx


def test_camera_management_registers_unique_encrypted_rtsp_sources(tmp_path: Path) -> None:
    service, repository, mediamtx = _service(tmp_path)

    first = service.register_camera(
        name="Entrance",
        location="Lobby",
        rtsp_url="rtsp://camera-one.local:554/stream?profile=main",
        username="operator",
        password="secret",
        source_on_demand=True,
    )
    second = service.register_camera(
        name="Parking",
        location=None,
        rtsp_url="rtsp://camera-two.local/stream",
        username=None,
        password=None,
        source_on_demand=True,
    )

    source = repository.get_camera_source(first.id)
    assert first.stream_path.startswith("cam-")
    assert first.stream_path != second.stream_path
    assert first.provisioning_status is CameraProvisioningStatus.ACTIVE
    assert first.rtsp_endpoint == "rtsp://camera-one.local:554/stream"
    assert source is not None and source.ciphertext is not None
    assert "secret" not in source.ciphertext
    assert mediamtx.paths[first.stream_path] == (
        "rtsp://operator:secret@camera-one.local:554/stream?profile=main",
        True,
    )


def test_camera_management_disables_and_deletes_dynamic_path(tmp_path: Path) -> None:
    service, _, mediamtx = _service(tmp_path)
    camera = service.register_camera(
        name="Entrance",
        location=None,
        rtsp_url="rtsp://camera.local/stream",
        username=None,
        password=None,
        source_on_demand=True,
    )

    disabled = service.update_camera(camera.id, {"enabled": False})
    deleted = service.delete_camera(camera.id)

    assert disabled.provisioning_status is CameraProvisioningStatus.DISABLED
    assert mediamtx.deleted == [camera.stream_path, camera.stream_path]
    assert deleted is True

