"""Camera registration orchestration and MediaMTX reconciliation."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping
from threading import Event, Thread
from typing import Any, Protocol
from uuid import uuid4

from cctv.cameras.credentials import CameraCredentialCipher, prepare_rtsp_source
from cctv.cameras.mediamtx import MediaMtxApiError, MediaMtxClient, MediaMtxPathStatus
from cctv.db import (
    CameraProvisioningStatus,
    CameraRecord,
    CameraRepository,
    CameraSourceRecord,
)

logger = logging.getLogger(__name__)


class CameraProvisioningError(RuntimeError):
    """A sanitized failure tied to a persisted camera registration."""

    def __init__(self, camera_id: int, message: str) -> None:
        self.camera_id = camera_id
        super().__init__(message)


class CameraManager(Protocol):
    def register_camera(
        self,
        *,
        name: str,
        location: str | None,
        rtsp_url: str,
        username: str | None,
        password: str | None,
        source_on_demand: bool,
    ) -> CameraRecord: ...

    def update_camera(
        self,
        camera_id: int,
        changes: Mapping[str, Any],
        *,
        rtsp_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        source_on_demand: bool | None = None,
    ) -> CameraRecord: ...

    def delete_camera(self, camera_id: int) -> bool: ...

    def sync_camera(self, camera_id: int) -> CameraRecord: ...

    def get_live_status(self, camera_id: int) -> MediaMtxPathStatus | None: ...


class CameraManagementService:
    """Keep encrypted SQLite registrations and MediaMTX paths consistent."""

    def __init__(
        self,
        repository: CameraRepository,
        cipher: CameraCredentialCipher,
        mediamtx: MediaMtxClient,
    ) -> None:
        self.repository = repository
        self.cipher = cipher
        self.mediamtx = mediamtx

    def register_camera(
        self,
        *,
        name: str,
        location: str | None,
        rtsp_url: str,
        username: str | None,
        password: str | None,
        source_on_demand: bool,
    ) -> CameraRecord:
        prepared = prepare_rtsp_source(rtsp_url, username=username, password=password)
        ciphertext = self.cipher.encrypt(prepared.source_url)
        record = self._create_with_unique_path(
            name=name,
            location=location,
            endpoint=prepared.endpoint,
            ciphertext=ciphertext,
            source_on_demand=source_on_demand,
        )
        return self._sync_or_record_error(record.id)

    def update_camera(
        self,
        camera_id: int,
        changes: Mapping[str, Any],
        *,
        rtsp_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        source_on_demand: bool | None = None,
    ) -> CameraRecord:
        current = self.repository.get_camera(camera_id)
        if current is None:
            raise LookupError(f"camera does not exist: {camera_id}")
        if changes:
            current = self.repository.update_camera(camera_id, changes)
        source_changed = rtsp_url is not None
        if not source_changed and (username is not None or password is not None):
            raise ValueError("rtsp_url is required when changing the RTSP account")
        if source_changed:
            prepared = prepare_rtsp_source(rtsp_url or "", username=username, password=password)
            current = self.repository.update_camera_source(
                camera_id,
                rtsp_endpoint=prepared.endpoint,
                rtsp_source_ciphertext=self.cipher.encrypt(prepared.source_url),
                source_on_demand=(
                    source_on_demand
                    if source_on_demand is not None
                    else current.source_on_demand
                ),
            )
        elif source_on_demand is not None and current.credentials_configured:
            source = self.repository.get_camera_source(camera_id)
            if source is None or source.ciphertext is None:
                raise RuntimeError("managed camera source is missing")
            current = self.repository.update_camera_source(
                camera_id,
                rtsp_endpoint=current.rtsp_endpoint or "rtsp://configured",
                rtsp_source_ciphertext=source.ciphertext,
                source_on_demand=source_on_demand,
            )
        if current.credentials_configured:
            return self._sync_or_record_error(camera_id)
        return current

    def delete_camera(self, camera_id: int) -> bool:
        source = self.repository.get_camera_source(camera_id)
        if source is None:
            return False
        if source.ciphertext is not None:
            try:
                self.mediamtx.delete_path(source.stream_path)
            except MediaMtxApiError as error:
                self._record_error(camera_id, error)
                raise CameraProvisioningError(
                    camera_id,
                    "MediaMTX 경로 삭제에 실패해 카메라 등록을 유지했습니다.",
                ) from error
        return self.repository.delete_camera(camera_id)

    def sync_camera(self, camera_id: int) -> CameraRecord:
        record = self.repository.get_camera(camera_id)
        if record is None:
            raise LookupError(f"camera does not exist: {camera_id}")
        if not record.credentials_configured:
            return record
        return self._sync_or_record_error(camera_id)

    def reconcile_all(self) -> None:
        for source in self.repository.list_managed_camera_sources():
            try:
                self._sync_or_record_error(source.camera_id)
            except (CameraProvisioningError, LookupError):
                continue

    def get_live_status(self, camera_id: int) -> MediaMtxPathStatus | None:
        record = self.repository.get_camera(camera_id)
        if record is None:
            raise LookupError(f"camera does not exist: {camera_id}")
        try:
            return self.mediamtx.get_path_status(record.stream_path)
        except MediaMtxApiError as error:
            raise CameraProvisioningError(
                camera_id,
                "MediaMTX에서 카메라 연결 상태를 확인할 수 없습니다.",
            ) from error

    def close(self) -> None:
        self.mediamtx.close()

    def _create_with_unique_path(
        self,
        *,
        name: str,
        location: str | None,
        endpoint: str,
        ciphertext: str,
        source_on_demand: bool,
    ) -> CameraRecord:
        for _ in range(5):
            stream_path = f"cam-{uuid4().hex[:12]}"
            try:
                return self.repository.create_camera(
                    name=name,
                    location=location,
                    stream_path=stream_path,
                    rtsp_endpoint=endpoint,
                    rtsp_source_ciphertext=ciphertext,
                    source_on_demand=source_on_demand,
                    provisioning_status=CameraProvisioningStatus.PENDING,
                )
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("could not allocate a unique camera stream path")

    def _sync_or_record_error(self, camera_id: int) -> CameraRecord:
        source = self.repository.get_camera_source(camera_id)
        if source is None:
            raise LookupError(f"camera does not exist: {camera_id}")
        try:
            return self._sync_source(source)
        except (MediaMtxApiError, ValueError) as error:
            self._record_error(camera_id, error)
            raise CameraProvisioningError(
                camera_id,
                "카메라는 저장했지만 MediaMTX 경로 동기화에 실패했습니다.",
            ) from error

    def _sync_source(self, source: CameraSourceRecord) -> CameraRecord:
        if source.ciphertext is None:
            record = self.repository.get_camera(source.camera_id)
            if record is None:
                raise LookupError(f"camera does not exist: {source.camera_id}")
            return record
        if not source.enabled:
            self.mediamtx.delete_path(source.stream_path)
            return self.repository.set_provisioning_result(
                source.camera_id,
                provisioning_status=CameraProvisioningStatus.DISABLED,
            )
        source_url = self.cipher.decrypt(source.ciphertext)
        self.mediamtx.upsert_rtsp_path(
            stream_path=source.stream_path,
            source_url=source_url,
            source_on_demand=source.source_on_demand,
        )
        return self.repository.set_provisioning_result(
            source.camera_id,
            provisioning_status=CameraProvisioningStatus.ACTIVE,
        )

    def _record_error(self, camera_id: int, error: Exception) -> None:
        error_name = type(error).__name__
        self.repository.set_provisioning_result(
            camera_id,
            provisioning_status=CameraProvisioningStatus.ERROR,
            error=error_name,
        )
        logger.warning(
            "Camera path synchronization failed",
            extra={
                "event": "camera_path_sync_failed",
                "camera_id": camera_id,
                "error_type": error_name,
            },
        )


class CameraReconciler:
    """Periodically restore dynamic paths after a MediaMTX restart."""

    def __init__(self, manager: CameraManagementService, *, interval_seconds: float) -> None:
        self.manager = manager
        self.interval_seconds = interval_seconds
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run, name="camera-reconciler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.interval_seconds + 1, 10))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.manager.reconcile_all()
            except Exception:
                logger.exception(
                    "Camera reconciliation cycle failed",
                    extra={"event": "camera_reconciliation_failed"},
                )
            self._stop_event.wait(self.interval_seconds)


__all__ = [
    "CameraManagementService",
    "CameraManager",
    "CameraProvisioningError",
    "CameraReconciler",
]
