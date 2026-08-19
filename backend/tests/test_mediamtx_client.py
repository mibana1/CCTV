import json

import httpx

from cctv.cameras import MediaMtxClient


def test_mediamtx_client_creates_dynamic_rtsp_path() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"status": "ok"})

    client = MediaMtxClient(
        "http://mediamtx:9997",
        transport=httpx.MockTransport(handler),
    )

    client.upsert_rtsp_path(
        stream_path="cam-123",
        source_url="rtsp://user:password@camera.local/stream",
        source_on_demand=True,
    )

    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].url.path == "/v3/config/paths/add/cam-123"
    assert json.loads(requests[1].content) == {
        "source": "rtsp://user:password@camera.local/stream",
        "sourceOnDemand": True,
        "rtspTransport": "tcp",
    }


def test_mediamtx_client_patches_existing_path_and_reads_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/v3/paths/get/"):
            return httpx.Response(200, json={"online": True, "available": True})
        return httpx.Response(200, json={"status": "ok"})

    client = MediaMtxClient(
        "http://mediamtx:9997",
        transport=httpx.MockTransport(handler),
    )

    client.upsert_rtsp_path(
        stream_path="cam-123",
        source_url="rtsp://camera.local/stream",
        source_on_demand=False,
    )
    status = client.get_path_status("cam-123")

    assert [request.method for request in requests] == ["GET", "PATCH", "GET"]
    assert requests[1].url.path == "/v3/config/paths/patch/cam-123"
    assert status is not None
    assert status.online is True
    assert status.available is True


def test_mediamtx_client_skips_patch_when_configuration_is_unchanged() -> None:
    requests: list[httpx.Request] = []
    expected = {
        "source": "rtsp://camera.local/stream",
        "sourceOnDemand": True,
        "rtspTransport": "tcp",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=expected)

    client = MediaMtxClient(
        "http://mediamtx:9997",
        transport=httpx.MockTransport(handler),
    )

    client.upsert_rtsp_path(
        stream_path="cam-123",
        source_url="rtsp://camera.local/stream",
        source_on_demand=True,
    )

    assert [request.method for request in requests] == ["GET"]


def test_mediamtx_client_treats_missing_delete_as_success() -> None:
    client = MediaMtxClient(
        "http://mediamtx:9997",
        transport=httpx.MockTransport(lambda _: httpx.Response(404)),
    )

    client.delete_path("cam-missing")
