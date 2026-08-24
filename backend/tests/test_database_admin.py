import shutil
from contextlib import closing
from pathlib import Path

import pytest

from cctv.db import connect_database, initialize_database
from cctv.db.admin import check_database, create_online_backup


def test_online_backup_can_be_restored_and_passes_full_integrity_check(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime" / "cctv.db"
    initialize_database(source)
    with closing(connect_database(source)) as connection, connection:
        connection.execute("INSERT INTO cameras (name) VALUES ('Lobby')")

    backup = create_online_backup(source, tmp_path / "backups" / "cctv.db")
    restored = tmp_path / "restore" / "cctv.db"
    restored.parent.mkdir(parents=True)
    shutil.copy2(backup.backup_path, restored)
    restored_check = check_database(restored)
    with closing(connect_database(restored)) as connection:
        camera_count = connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]

    assert backup.integrity.integrity_ok is True
    assert backup.integrity.integrity_messages == ("ok",)
    assert backup.backup_bytes > 0
    assert restored_check.integrity_ok is True
    assert restored_check.schema_version == 21
    assert camera_count == 1


def test_online_backup_refuses_source_or_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "cctv.db"
    initialize_database(source)

    with pytest.raises(ValueError, match="differ"):
        create_online_backup(source, source)
    with pytest.raises(FileNotFoundError):
        create_online_backup(tmp_path / "missing.db", tmp_path / "missing-backup.db")
    assert not (tmp_path / "missing.db").exists()
    destination = tmp_path / "existing.db"
    destination.write_bytes(b"do not replace")
    with pytest.raises(FileExistsError):
        create_online_backup(source, destination)
    assert destination.read_bytes() == b"do not replace"
