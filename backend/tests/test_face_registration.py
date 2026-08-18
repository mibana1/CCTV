import sqlite3
from pathlib import Path

import pytest

from cctv.db import IdentityEmbeddingInput, IdentityRepository, initialize_database
from cctv.identity import (
    SFACE_EMBEDDING_DIMENSION,
    ExtractedFaceEmbedding,
    FaceEmbeddingModelMetadata,
    FacePhotoError,
    FaceRegistrationError,
    FaceRegistrationService,
    discover_registration_photos,
)


class _Extractor:
    metadata = FaceEmbeddingModelMetadata()

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def extract_file(self, photo_path: str | Path) -> ExtractedFaceEmbedding:
        path = Path(photo_path)
        if path.name == self.fail_on:
            raise FacePhotoError(f"invalid face: {path.name}")
        index = int(path.stem.split("_")[0]) - 1
        vector = [0.0] * SFACE_EMBEDDING_DIMENSION
        vector[index] = 1.0
        return ExtractedFaceEmbedding(tuple(vector), 0.9 + index * 0.01)


def _photos(directory: Path, count: int) -> None:
    directory.mkdir(parents=True)
    for index in range(1, count + 1):
        (directory / f"{index:02d}_photo.jpg").write_bytes(b"test")


def _repository(tmp_path: Path) -> IdentityRepository:
    database_path = tmp_path / "cctv.db"
    initialize_database(database_path)
    repository = IdentityRepository(database_path)
    repository.create_identity(
        identity_id="person-1",
        external_id="EMP-001",
        display_name="Person",
    )
    return repository


def test_face_registration_stores_three_photo_embeddings_atomically(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    photo_root = tmp_path / "identity-images"
    photo_directory = photo_root / "EMP-001"
    _photos(photo_directory, 3)

    result = FaceRegistrationService(
        repository,
        _Extractor(),
        photo_root=photo_root,
    ).register("person-1")

    assert result.photo_directory == photo_directory.resolve()
    assert [item.photo_name for item in result.items] == [
        "01_photo.jpg",
        "02_photo.jpg",
        "03_photo.jpg",
    ]
    assert all(item.embedding.model_name == "opencv_sface" for item in result.items)
    assert all(item.embedding.model_version == "2021dec" for item in result.items)
    assert all(item.embedding.dimension == SFACE_EMBEDDING_DIMENSION for item in result.items)
    assert repository.list_embeddings("person-1").total == 3


def test_face_registration_does_not_store_partial_results(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    photo_directory = tmp_path / "photos"
    _photos(photo_directory, 3)
    service = FaceRegistrationService(
        repository, _Extractor(fail_on="03_photo.jpg"), photo_root=tmp_path
    )

    with pytest.raises(FacePhotoError, match="03_photo.jpg"):
        service.register("person-1", photo_directory=photo_directory)

    assert repository.list_embeddings("person-1").total == 0


@pytest.mark.parametrize("count", [2, 6])
def test_face_registration_requires_three_to_five_supported_photos(
    tmp_path: Path,
    count: int,
) -> None:
    photo_directory = tmp_path / "photos"
    _photos(photo_directory, count)

    with pytest.raises(FaceRegistrationError, match="requires 3 to 5"):
        discover_registration_photos(photo_directory)


def test_identity_repository_rolls_back_duplicate_embedding_batch(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    duplicate = IdentityEmbeddingInput(model_name="opencv_sface", vector=[1, 0])

    with pytest.raises(sqlite3.IntegrityError):
        repository.add_embeddings("person-1", (duplicate, duplicate))

    assert repository.list_embeddings("person-1").total == 0
