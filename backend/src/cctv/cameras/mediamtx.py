"""Small, credential-safe client for the MediaMTX control API."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx


class MediaMtxApiError(RuntimeError):
    """Sanitized MediaMTX failure safe for logs and API responses."""

    def __init__(self, operation: str, status_code: int | None = None) -> None:
        self.operation = operation
        self.status_code = status_code
        detail = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"MediaMTX {operation} failed{detail}")


@dataclass(frozen=True, slots=True)
class MediaMtxPathStatus:
    online: bool
    available: bool


class MediaMtxClient:
    """Manage dynamic path configurations through an internal-only endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds, transport=transport)

    def upsert_rtsp_path(
        self,
        *,
        stream_path: str,
        source_url: str,
        source_on_demand: bool,
    ) -> None:
        encoded_path = quote(stream_path, safe="")
        config_path = f"/v3/config/paths/get/{encoded_path}"
        current = self._request("GET", config_path, operation="path lookup", allow_not_found=True)
        payload = {
            "source": source_url,
            "sourceOnDemand": source_on_demand,
            "rtspTransport": "tcp",
        }
        if current.status_code == 404:
            response = self._request(
                "POST",
                f"/v3/config/paths/add/{encoded_path}",
                operation="path creation",
                json=payload,
            )
        else:
            try:
                current_config = current.json()
            except ValueError as error:
                raise MediaMtxApiError("path configuration decoding", current.status_code) from error
            if all(current_config.get(key) == value for key, value in payload.items()):
                return
            response = self._request(
                "PATCH",
                f"/v3/config/paths/patch/{encoded_path}",
                operation="path update",
                json=payload,
            )
        if response.status_code != 200:
            raise MediaMtxApiError("path synchronization", response.status_code)

    def delete_path(self, stream_path: str) -> None:
        encoded_path = quote(stream_path, safe="")
        response = self._request(
            "DELETE",
            f"/v3/config/paths/delete/{encoded_path}",
            operation="path deletion",
            allow_not_found=True,
        )
        if response.status_code not in {200, 404}:
            raise MediaMtxApiError("path deletion", response.status_code)

    def get_path_status(self, stream_path: str) -> MediaMtxPathStatus | None:
        encoded_path = quote(stream_path, safe="")
        response = self._request(
            "GET",
            f"/v3/paths/get/{encoded_path}",
            operation="runtime path lookup",
            allow_not_found=True,
        )
        if response.status_code == 404:
            return None
        try:
            body = response.json()
        except ValueError as error:
            raise MediaMtxApiError("runtime path decoding", response.status_code) from error
        return MediaMtxPathStatus(
            online=bool(body.get("online", body.get("ready", False))),
            available=bool(body.get("available", body.get("ready", False))),
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        allow_not_found: bool = False,
        json: dict[str, object] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", json=json)
        except httpx.HTTPError as error:
            raise MediaMtxApiError(operation) from error
        if response.status_code == 404 and allow_not_found:
            return response
        if response.status_code >= 400:
            raise MediaMtxApiError(operation, response.status_code)
        return response


__all__ = ["MediaMtxApiError", "MediaMtxClient", "MediaMtxPathStatus"]
