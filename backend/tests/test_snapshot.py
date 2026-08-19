from pathlib import Path

import cv2
import numpy as np
import pytest

from cctv.media import (
    DecodedFrame,
    SnapshotEncodingError,
    SnapshotOverlay,
    SnapshotWriteError,
    SnapshotWriter,
    build_snapshot_run_directory,
    render_snapshot_annotations,
    sample_index_from_snapshot_name,
)


def make_frame(
    *,
    source_index: int = 2,
    sample_index: int = 1,
    timestamp_seconds: float = 0.5,
) -> DecodedFrame:
    image = np.full((24, 32, 3), 127, dtype=np.uint8)
    return DecodedFrame(
        source_index=source_index,
        sample_index=sample_index,
        timestamp_seconds=timestamp_seconds,
        image=image,
    )


def test_snapshot_writer_saves_atomic_jpeg_with_deterministic_name(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path / "run", jpeg_quality=95)

    record = writer.save(make_frame())

    assert record.path.name == "sample_000001_frame_000000002_t000000000500ms.jpg"
    assert record.size_bytes > 0
    assert writer.saved_count == 1
    assert writer.first_path == record.path
    assert writer.last_path == record.path
    assert not list(writer.output_dir.glob("*.tmp"))
    decoded = cv2.imread(str(record.path))
    assert decoded is not None
    assert decoded.shape == (24, 32, 3)
    assert float(decoded.mean()) == pytest.approx(127, abs=2)


def test_snapshot_writer_tracks_first_and_last_paths(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path / "run")

    first = writer.save(make_frame(source_index=0, sample_index=0, timestamp_seconds=0))
    second = writer.save(make_frame(source_index=4, sample_index=2, timestamp_seconds=1))

    assert writer.saved_count == 2
    assert writer.first_path == first.path
    assert writer.last_path == second.path
    assert sorted(path.name for path in writer.output_dir.glob("*.jpg")) == [
        first.path.name,
        second.path.name,
    ]


def test_snapshot_writer_refuses_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    writer = SnapshotWriter(output_dir)

    with pytest.raises(SnapshotWriteError, match="already exists"):
        writer.save(make_frame())


def test_snapshot_writer_reports_encoding_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, np.array([])))
    writer = SnapshotWriter(tmp_path / "run")

    with pytest.raises(SnapshotEncodingError, match="encoding failed"):
        writer.save(make_frame())

    assert not writer.output_dir.exists()


@pytest.mark.parametrize("jpeg_quality", [0, 101])
def test_snapshot_writer_rejects_invalid_jpeg_quality(jpeg_quality: int) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        SnapshotWriter("snapshots", jpeg_quality=jpeg_quality)


def test_build_snapshot_run_directory_is_unique_and_source_named(tmp_path: Path) -> None:
    first = build_snapshot_run_directory(tmp_path, "보안 카메라 1.mp4")
    second = build_snapshot_run_directory(tmp_path, "보안 카메라 1.mp4")

    assert first.parent == tmp_path.resolve()
    assert first.name.startswith("보안_카메라_1_")
    assert first != second
    assert not first.exists()
    assert not second.exists()


def test_snapshot_annotation_draws_tracking_and_person_labels(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path / "run", jpeg_quality=100)
    record = writer.save(make_frame())

    rendered = render_snapshot_annotations(
        record.path,
        (
            SnapshotOverlay(
                x1=3,
                y1=3,
                x2=28,
                y2=21,
                primary_label="person 0.91 | Track 27",
                secondary_label="Person 0dd07e6a | MATCHED",
                color=(42, 190, 80),
            ),
        ),
        jpeg_quality=100,
    )

    decoded = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
    original = cv2.imread(str(record.path))
    assert decoded is not None
    assert original is not None
    assert decoded.shape == original.shape
    assert not np.array_equal(decoded, original)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sample_000123_frame_000000456_t000000500ms.jpg", 123),
        ("sample.jpg", None),
        ("other_000123.jpg", None),
    ],
)
def test_sample_index_from_snapshot_name(name: str, expected: int | None) -> None:
    assert sample_index_from_snapshot_name(name) == expected
