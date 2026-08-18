import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import IdentityRepository, connect_database, initialize_database


def test_identity_repository_manages_profiles_and_filters(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = IdentityRepository(database_path)

    identity = repository.create_identity(
        identity_id="person-1",
        external_id="EMP-001",
        display_name="노주형",
        description="테스트 등록 인물",
        metadata={"department": "개발", "access_level": 2},
    )
    repository.create_identity(
        identity_id="person-2",
        external_id="visitor-1",
        display_name="방문자",
        enabled=False,
    )

    assert repository.get_identity(identity.id) == identity
    assert repository.list_identities(query="주형", enabled=True).items == (identity,)
    assert repository.list_identities(query="emp-001").items == (identity,)
    assert repository.list_identities(page=1, limit=1).total == 2

    updated = repository.update_identity(
        identity.id,
        {
            "description": None,
            "external_id": None,
            "metadata": {"department": "플랫폼"},
            "enabled": False,
        },
    )
    assert updated.description is None
    assert updated.external_id is None
    assert updated.metadata == {"department": "플랫폼"}
    assert updated.enabled is False

    assert repository.delete_identity(identity.id) is True
    assert repository.delete_identity(identity.id) is False
    assert repository.get_identity(identity.id) is None


def test_identity_repository_normalizes_and_loads_model_specific_vectors(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = IdentityRepository(database_path)
    repository.create_identity(identity_id="enabled", display_name="Enabled")
    repository.create_identity(identity_id="disabled", display_name="Disabled", enabled=False)

    embedding = repository.add_embedding(
        "enabled",
        embedding_id="embedding-1",
        model_name="arcface",
        model_version="1.0",
        vector=[3, 4, 0],
        source_reference="registration-image-1",
        quality_score=0.95,
    )
    repository.add_embedding(
        "disabled",
        model_name="arcface",
        model_version="1.0",
        vector=[0, 1, 0],
    )
    repository.add_embedding(
        "enabled",
        model_name="other-model",
        model_version="1.0",
        vector=[1, 0],
    )

    assert embedding.dimension == 3
    assert embedding.dtype == "float32"
    assert embedding.normalized is True
    assert len(embedding.embedding_sha256) == 64
    assert repository.list_embeddings(
        "enabled", model_name="ARCFACE", model_version="1.0"
    ).items == (embedding,)

    candidates = repository.list_embedding_vectors(
        model_name="ARCFACE",
        model_version="1.0",
        dimension=3,
    )
    assert len(candidates) == 1
    assert candidates[0].identity.id == "enabled"
    assert candidates[0].embedding == embedding
    assert candidates[0].vector == pytest.approx((0.6, 0.8, 0.0))

    all_candidates = repository.list_embedding_vectors(
        model_name="arcface",
        model_version="1.0",
        enabled_identities_only=False,
    )
    assert {candidate.identity.id for candidate in all_candidates} == {"enabled", "disabled"}


def test_identity_repository_enforces_registration_integrity(tmp_path: Path) -> None:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = IdentityRepository(database_path)
    repository.create_identity(
        identity_id="person-1",
        external_id="EMP-001",
        display_name="Person",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_identity(external_id="emp-001", display_name="Duplicate")
    with pytest.raises(LookupError):
        repository.add_embedding("missing", model_name="arcface", vector=[1, 0])
    with pytest.raises(ValueError, match="dimension"):
        repository.add_embedding("person-1", model_name="arcface", vector=[1])
    with pytest.raises(ValueError, match="finite"):
        repository.add_embedding("person-1", model_name="arcface", vector=[1, float("inf")])
    with pytest.raises(ValueError, match="non-zero"):
        repository.add_embedding("person-1", model_name="arcface", vector=[0, 0])
    with pytest.raises(ValueError, match="quality_score"):
        repository.add_embedding(
            "person-1",
            model_name="arcface",
            vector=[1, 0],
            quality_score=1.1,
        )

    repository.add_embedding(
        "person-1",
        embedding_id="embedding-1",
        model_name="arcface",
        vector=[1, 0],
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_embedding("person-1", model_name="arcface", vector=[2, 0])
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_embedding("person-1", model_name="ARCFACE", vector=[1, 0])

    assert repository.delete_identity("person-1") is True
    with closing(connect_database(database_path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM identity_embeddings").fetchone()[0]
    assert count == 0
