import sqlite3
from pathlib import Path

import numpy as np
import pytest

from findmyfit.config import Settings
from findmyfit.db.catalog_migration import audit_catalog, migrate_catalog
from findmyfit.errors import CatalogError


def _legacy_catalog(path: Path) -> None:
    vector = np.ones(3, dtype=np.float32)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE images (
                id VARCHAR PRIMARY KEY,
                file_path VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                hash VARCHAR
            );
            CREATE TABLE models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                embedding_dim INTEGER NOT NULL,
                description VARCHAR,
                UNIQUE(name, version)
            );
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id VARCHAR NOT NULL,
                model_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                dtype VARCHAR NOT NULL,
                created_at DATETIME,
                UNIQUE(image_id, model_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?)",
            ("item", "shoes/item.jpg", "shoes", "hash"),
        )
        connection.execute(
            "INSERT INTO models(name, version, embedding_dim) VALUES (?, ?, ?)",
            ("test", "v1", 256),
        )
        connection.execute(
            "INSERT INTO embeddings(image_id, model_id, vector, dim, dtype) "
            "VALUES (?, ?, ?, ?, ?)",
            ("item", 1, vector.tobytes(), 256, "float32"),
        )


def test_catalog_migration_is_dry_run_then_backed_up_and_idempotent(
    tmp_path: Path,
    monkeypatch,
):
    database_path = tmp_path / "catalog.db"
    _legacy_catalog(database_path)
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
        images_dir=tmp_path / "images",
    )
    monkeypatch.setattr(
        "findmyfit.db.catalog_migration.expected_model_metadata",
        lambda *_args: (3, "fingerprint"),
    )

    before, backup = migrate_catalog(settings, apply=False)
    assert backup is None
    assert not before.fingerprint_column
    assert before.missing_indexes == [
        "ix_embeddings_model_id",
        "ix_images_category",
    ]

    after, backup = migrate_catalog(settings, apply=True)
    assert backup is not None and backup.is_file()
    assert after.ready
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT embedding_dim, artifact_fingerprint FROM models"
        ).fetchone() == (3, "fingerprint")
        assert connection.execute("SELECT dim FROM embeddings").fetchone() == (3,)

    repeated, second_backup = migrate_catalog(settings, apply=True)
    assert repeated.ready
    assert second_backup is not None and second_backup.is_file()
    assert audit_catalog(settings).ready


def test_catalog_migration_rolls_back_incompatible_fingerprint(
    tmp_path: Path,
    monkeypatch,
):
    database_path = tmp_path / "catalog.db"
    _legacy_catalog(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "ALTER TABLE models ADD COLUMN artifact_fingerprint VARCHAR"
        )
        connection.execute(
            "UPDATE models SET artifact_fingerprint = 'different'"
        )
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        database_url=f"sqlite:///{database_path.as_posix()}",
        images_dir=tmp_path / "images",
    )
    monkeypatch.setattr(
        "findmyfit.db.catalog_migration.expected_model_metadata",
        lambda *_args: (3, "fingerprint"),
    )

    with pytest.raises(CatalogError, match="incompatible"):
        migrate_catalog(settings, apply=True)

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT embedding_dim, artifact_fingerprint FROM models"
        ).fetchone() == (256, "different")
        existing_indexes = {
            row[1]
            for table in ("images", "embeddings")
            for row in connection.execute(f"PRAGMA index_list({table})")
        }
    assert "ix_images_category" not in existing_indexes
    assert "ix_embeddings_model_id" not in existing_indexes
