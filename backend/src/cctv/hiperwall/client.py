"""Synchronous HiperInterface HTTP/XML client used by the durable worker."""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree

import httpx


class HiperwallRequestError(RuntimeError):
    """Sanitized HiperInterface failure safe for logs and the database."""

    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class HiperwallClient:
    """Send open and close commands without handling queue state."""

    def __init__(
        self,
        *,
        base_url: str,
        auth_mode: str = "none",
        user: str = "",
        token: str = "",
        timeout_seconds: float = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_mode = auth_mode
        self.user = user
        self._token = token
        if auth_mode not in {"none", "token"}:
            raise ValueError("Hiperwall live client supports none or token authentication")
        if auth_mode == "token" and (not user or not token):
            raise ValueError("Hiperwall token authentication requires user and token")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def hello(self) -> dict[str, Any]:
        try:
            response = self._client.get(f"{self.base_url}/hello")
        except httpx.HTTPError as error:
            raise HiperwallRequestError(
                "hiperwall_transport_error",
                "Hiperwall connection failed",
            ) from error
        if response.status_code < 200 or response.status_code >= 300:
            raise HiperwallRequestError(
                f"hiperwall_http_{response.status_code}",
                f"Hiperwall returned HTTP {response.status_code}",
                retryable=response.status_code >= 500,
            )
        return {"http_status": response.status_code, "hello": response.text.strip()}

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "open_source":
            xml = self._open_xml(request)
        elif operation == "restore_layout":
            xml = self._close_xml(request)
        else:
            raise HiperwallRequestError(
                "hiperwall_invalid_operation",
                "Unsupported Hiperwall operation",
                retryable=False,
            )
        response = self._post_xml(xml)
        return {
            "external_request_sent": True,
            "http_status": response.status_code,
            "operation": operation,
            "instance_id": request["instance_id"],
        }

    def _post_xml(self, xml: str) -> httpx.Response:
        try:
            response = self._client.post(
                f"{self.base_url}/xmlcommand",
                headers={"content-type": "application/xml; charset=utf-8"},
                content=xml.encode("utf-8"),
            )
        except httpx.TimeoutException as error:
            raise HiperwallRequestError(
                "hiperwall_timeout",
                "Hiperwall request timed out",
            ) from error
        except httpx.HTTPError as error:
            raise HiperwallRequestError(
                "hiperwall_transport_error",
                "Hiperwall request failed",
            ) from error
        if response.status_code < 200 or response.status_code >= 300:
            raise HiperwallRequestError(
                f"hiperwall_http_{response.status_code}",
                f"Hiperwall returned HTTP {response.status_code}",
                retryable=response.status_code >= 500,
            )
        if _contains_xml_error(response.text):
            raise HiperwallRequestError(
                "hiperwall_business_error",
                "Hiperwall XML response reported a business error",
                retryable=False,
            )
        return response

    def _root(self) -> ElementTree.Element:
        root = ElementTree.Element("Commands")
        if self.auth_mode == "token":
            auth = ElementTree.SubElement(root, "auth", {"type": "token"})
            ElementTree.SubElement(auth, "user").text = self.user
            ElementTree.SubElement(auth, "token").text = self._token
        return root

    def _open_xml(self, request: dict[str, Any]) -> str:
        target = request.get("target")
        if not isinstance(target, dict):
            raise HiperwallRequestError(
                "hiperwall_invalid_target",
                "Hiperwall target is missing",
                retryable=False,
            )
        selector = target.get("selector")
        value = target.get("value")
        if selector not in {"name", "uuid"} or not isinstance(value, str) or not value:
            raise HiperwallRequestError(
                "hiperwall_invalid_target",
                "Hiperwall content target is invalid",
                retryable=False,
            )
        root = self._root()
        command = ElementTree.SubElement(root, "command", {"type": "open"})
        ElementTree.SubElement(command, selector).text = value
        ElementTree.SubElement(command, "id").text = _instance_id(request)
        zone_id = target.get("zone_id")
        if isinstance(zone_id, str) and zone_id:
            ElementTree.SubElement(command, "zone").text = zone_id
            self._add_layout(command, zone_id, target.get("layout"))
        label = request.get("label")
        if isinstance(label, str) and label:
            ElementTree.SubElement(command, "label").text = label[:256]
            ElementTree.SubElement(command, "showlabel").text = "true"
        return _xml_text(root)

    def _close_xml(self, request: dict[str, Any]) -> str:
        root = self._root()
        command = ElementTree.SubElement(root, "command", {"type": "close"})
        ElementTree.SubElement(command, "id").text = _instance_id(request)
        return _xml_text(root)

    @staticmethod
    def _add_layout(
        command: ElementTree.Element,
        zone_id: str,
        layout: object,
    ) -> None:
        if not isinstance(layout, dict):
            return
        if layout.get("mode") == "pixels":
            for tag, field in (
                ("x", "x"),
                ("y", "y"),
                ("boundsw", "width"),
                ("boundsh", "height"),
            ):
                ElementTree.SubElement(command, tag).text = str(layout[field])
            ElementTree.SubElement(command, "boundsfill").text = "1"
            return
        left = float(layout["x"]) / 100
        top = float(layout["y"]) / 100
        right = (float(layout["x"]) + float(layout["width"])) / 100
        bottom = (float(layout["y"]) + float(layout["height"])) / 100
        values = ",".join(_fraction(value) for value in (left, top, right, bottom))
        ElementTree.SubElement(command, "position").text = f"{zone_id}:{values}"


def _instance_id(request: dict[str, Any]) -> str:
    value = request.get("instance_id")
    if not isinstance(value, str) or not value:
        raise HiperwallRequestError(
            "hiperwall_invalid_instance",
            "Hiperwall instance ID is missing",
            retryable=False,
        )
    return value


def _fraction(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _xml_text(root: ElementTree.Element) -> str:
    return ElementTree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
    ).decode("utf-8")


def _contains_xml_error(value: str) -> bool:
    markup = re.sub(r"<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>", "", value)
    return bool(
        re.search(
            r"<\s*(?:[A-Za-z_][\w.-]*:)?Error\b[^>]*>",
            markup,
            re.IGNORECASE,
        )
    )


__all__ = ["HiperwallClient", "HiperwallRequestError"]
