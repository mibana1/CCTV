from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.hiperwall import HiperwallClient, HiperwallRequestError
from cctv.main import create_app

CONTENT_XML = """\
<Objects>
  <Object type="Streamer">
    <name>Lobby &amp; Entrance</name>
    <type>Streamer</type>
    <width>1920</width>
    <height>1080</height>
    <uuid>source-123</uuid>
    <label>Lobby camera</label>
    <Instance>
      <id>open-1</id>
      <position>-10.5,20</position>
      <size>1280,720</size>
      <rotation>0</rotation>
      <transparency>1</transparency>
      <rgb>1,0.5,1</rgb>
      <bw>0</bw>
      <mosaic>0</mosaic>
      <layer>2</layer>
      <showlabel>true</showlabel>
      <borderRGB>00ffff</borderRGB>
      <bordervis>90</bordervis>
      <audio>75,muted</audio>
    </Instance>
  </Object>
  <Object type="Image">
    <name>floor-plan.png</name>
    <type>Image</type>
  </Object>
</Objects>
"""

WALLS_XML = """\
<Zones>
  <Zone>
    <name>Alert Zone</name>
    <id>zone-abc</id>
    <left>-1920</left>
    <top>0</top>
    <width>1920</width>
    <height>1080</height>
    <color>#FF3333</color>
    <zonegridh>2</zonegridh>
    <zonegridv>1</zonegridv>
  </Zone>
</Zones>
"""


def _settings(tmp_path: Path, *, base_url: str | None = "http://hiperwall-host:8000") -> Settings:
    return Settings(
        _env_file=None,
        app_mode="dry_run",
        hiperwall_base_url=base_url,
        database_path=tmp_path / "cctv.db",
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "cctv.jsonl",
    )


def _client(requests: list[httpx.Request]) -> HiperwallClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/hello":
            return httpx.Response(200, text="Hiperwall,2025R1,Token,Default")
        body = request.content.decode("utf-8")
        if '<action type="list"' in body:
            return httpx.Response(200, text=CONTENT_XML)
        if '<action type="walls"' in body:
            return httpx.Response(200, text=WALLS_XML)
        return httpx.Response(400, text="unexpected request")

    return HiperwallClient(
        base_url="http://hiperwall-host:8000",
        auth_mode="token",
        user="cctv_bridge",
        token="secret-token",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_client_inventory_returns_contents_instances_and_zone_geometry() -> None:
    requests: list[httpx.Request] = []
    client = _client(requests)

    inventory = client.inventory()

    assert inventory["hello"] == "Hiperwall,2025R1,Token,Default"
    assert inventory["contents"][0] == {
        "name": "Lobby & Entrance",
        "type": "Streamer",
        "uuid": "source-123",
        "label": "Lobby camera",
        "width": 1920.0,
        "height": 1080.0,
        "zone_id": None,
        "instances": [
            {
                "id": "open-1",
                "position": [-10.5, 20.0],
                "size": [1280.0, 720.0],
                "rotation": 0.0,
                "transparency": 1.0,
                "rgb": [1.0, 0.5, 1.0],
                "black_and_white": 0.0,
                "mosaic": 0.0,
                "layer": 2.0,
                "show_label": True,
                "border_rgb": "00ffff",
                "border_visibility": 90.0,
                "audio_volume": 75.0,
                "audio_muted": True,
            }
        ],
    }
    assert inventory["contents"][1]["uuid"] is None
    assert inventory["zones"] == [
        {
            "id": "zone-abc",
            "name": "Alert Zone",
            "left": -1920.0,
            "top": 0.0,
            "width": 1920.0,
            "height": 1080.0,
            "color": "#FF3333",
            "grid_horizontal": 2,
            "grid_vertical": 1,
        }
    ]
    assert inventory["walls"] == []
    assert len(requests) == 3
    assert all("secret-token" not in request.url.query.decode() for request in requests)
    assert all(
        "<token>secret-token</token>" in request.content.decode("utf-8")
        for request in requests[1:]
    )


def test_inventory_api_uses_configured_client_and_excludes_missing_values(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    hiperwall_client = _client(requests)

    with TestClient(create_app(_settings(tmp_path), hiperwall_client=hiperwall_client)) as client:
        response = client.get("/hiperwall/inventory")

    assert response.status_code == 200
    body = response.json()
    assert body["contents"][0]["uuid"] == "source-123"
    assert body["contents"][0]["instances"][0]["id"] == "open-1"
    assert body["contents"][1] == {
        "name": "floor-plan.png",
        "type": "Image",
        "instances": [],
    }
    assert body["zones"][0]["id"] == "zone-abc"


def test_inventory_api_requires_hyperwall_base_url(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, base_url=None))) as client:
        response = client.get("/hiperwall/inventory")

    assert response.status_code == 503
    assert response.json()["detail"] == "HIPERWALL_BASE_URL이 설정되지 않았습니다."


def test_inventory_api_reports_invalid_hyperwall_url_as_configuration_error(
    tmp_path: Path,
) -> None:
    with TestClient(
        create_app(_settings(tmp_path, base_url="http://hiperwall-host:port"))
    ) as client:
        response = client.get("/hiperwall/inventory")

    assert response.status_code == 503
    assert response.json()["detail"] == "Hiperwall 목록 조회 실패: Hiperwall URL is invalid"


def test_client_inventory_rejects_invalid_xml() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/hello":
            return httpx.Response(200, text="Hiperwall,2025R1,None,Default")
        return httpx.Response(200, text="<Objects>")

    client = HiperwallClient(
        base_url="http://hiperwall-host:8000",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(HiperwallRequestError) as captured:
        client.inventory()

    assert captured.value.code == "hiperwall_invalid_xml"
