from pathlib import Path

from fastapi.testclient import TestClient

from cctv.core.settings import Settings
from cctv.main import create_app


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_path=tmp_path / "runtime" / "cctv.db",
        model_path=tmp_path / "model.onnx",
        log_path=tmp_path / "runtime" / "cctv.jsonl",
    )


def test_identities_api_manages_profiles_and_pagination(tmp_path: Path) -> None:
    body = {
        "external_id": "EMP-001",
        "display_name": "노주형",
        "description": "등록 테스트",
        "metadata": {"department": "개발"},
    }
    with TestClient(create_app(_settings(tmp_path))) as client:
        created = client.post("/identities", json=body)
        identity_id = created.json()["id"]
        duplicate = client.post("/identities", json={**body, "external_id": "emp-001"})
        listed = client.get(
            "/identities",
            params={"query": "EMP", "enabled": True, "page": 1, "limit": 1},
        )
        detail = client.get(f"/identities/{identity_id}")
        updated = client.patch(
            f"/identities/{identity_id}",
            json={"description": None, "metadata": {"department": "플랫폼"}, "enabled": False},
        )
        deleted = client.delete(f"/identities/{identity_id}")
        missing = client.get(f"/identities/{identity_id}")

    assert created.status_code == 201
    assert created.json()["display_name"] == body["display_name"]
    assert duplicate.status_code == 409
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["has_next"] is False
    assert detail.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["description"] is None
    assert updated.json()["metadata"] == {"department": "플랫폼"}
    assert updated.json()["enabled"] is False
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_identities_api_manages_embeddings_without_exposing_vectors(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        identity = client.post("/identities", json={"display_name": "Person"}).json()
        path = f"/identities/{identity['id']}/embeddings"
        body = {
            "model_name": "arcface",
            "model_version": "1.0",
            "vector": [3, 4, 0],
            "source_reference": "registration-image-1",
            "quality_score": 0.9,
        }
        created = client.post(path, json=body)
        duplicate = client.post(path, json=body)
        listed = client.get(path, params={"model_name": "ARCFACE", "model_version": "1.0"})
        embedding_id = created.json()["id"]
        deleted = client.delete(f"{path}/{embedding_id}")
        missing = client.delete(f"{path}/{embedding_id}")

    assert created.status_code == 201
    assert created.json()["dimension"] == 3
    assert created.json()["normalized"] is True
    assert "vector" not in created.json()
    assert duplicate.status_code == 409
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert "vector" not in listed.json()["items"][0]
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_identities_api_rejects_invalid_registration_requests(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        identity_id = client.post("/identities", json={"display_name": "Person"}).json()["id"]
        path = f"/identities/{identity_id}/embeddings"

        empty_patch = client.patch(f"/identities/{identity_id}", json={})
        null_name = client.patch(f"/identities/{identity_id}", json={"display_name": None})
        whitespace_name = client.post("/identities", json={"display_name": "   "})
        whitespace_query = client.get("/identities", params={"query": "   "})
        zero_vector = client.post(path, json={"model_name": "arcface", "vector": [0, 0]})
        short_vector = client.post(path, json={"model_name": "arcface", "vector": [1]})
        missing_identity = client.post(
            "/identities/missing/embeddings",
            json={"model_name": "arcface", "vector": [1, 0]},
        )
        missing_list = client.get("/identities/missing/embeddings")
        invalid_limit = client.get("/identities", params={"limit": 101})

    assert empty_patch.status_code == 422
    assert null_name.status_code == 422
    assert whitespace_name.status_code == 422
    assert whitespace_query.status_code == 422
    assert zero_vector.status_code == 422
    assert "non-zero" in zero_vector.json()["detail"]
    assert short_vector.status_code == 422
    assert missing_identity.status_code == 404
    assert missing_list.status_code == 404
    assert invalid_limit.status_code == 422
