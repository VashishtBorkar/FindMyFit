import json
from pathlib import Path

import numpy as np
import pytest

import findmyfit.retrieval.faiss_index as faiss_module
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.models import Base
from findmyfit.db.session import create_engine_for_url, create_session_factory
from findmyfit.errors import VectorIndexError
from findmyfit.retrieval.faiss_index import (
    FaissCatalogSearch,
    audit_faiss_indexes,
    build_faiss_indexes,
)
from findmyfit.retrieval.sqlite_search import SqliteLinearSearch
from findmyfit.storage.local_images import LocalImageStore


def _fixture(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}"
    Base.metadata.create_all(create_engine_for_url(database_url))
    repository = SqliteEmbeddingRepository(
        create_session_factory(database_url),
        LocalImageStore(tmp_path / "images"),
    )
    model_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="fingerprint",
    )
    records = [
        ("near", "shoes", "near", [1.0, 0.0]),
        ("duplicate", "shoes", "near", [1.0, 0.0]),
        ("hashless-a", "shoes", None, [0.8, 0.2]),
        ("hashless-b", "shoes", None, [0.7, 0.3]),
        ("far", "shoes", "far", [0.0, 1.0]),
        ("pants", "pants", "pants", [0.9, 0.1]),
    ]
    for item_id, category, image_hash, vector in records:
        normalized = np.asarray(vector, dtype=np.float32)
        normalized /= np.linalg.norm(normalized)
        repository.upsert_embedding(
            model_id=model_id,
            item_id=item_id,
            category=category,
            image_key=f"{category}/{item_id}.jpg",
            image_hash=image_hash,
            vector=normalized,
        )
    return repository


def test_faiss_exact_cosine_matches_sqlite_and_partitions_categories(
    tmp_path: Path,
):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    manifest = build_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="cosine",
    )
    assert set(manifest.categories) == {"pants", "shoes"}
    assert (root / "clip" / "v1" / "shoes.faiss").is_file()

    faiss_search = FaissCatalogSearch(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="cosine",
    )
    sqlite_search = SqliteLinearSearch(
        repository,
        model_name="clip",
        model_version="v1",
        metric="cosine",
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    faiss_hits = faiss_search.search(query, ["shoes", "pants"], 5)
    sqlite_hits = sqlite_search.search(query, ["shoes", "pants"], 5)
    assert [hit.item_id for hit in faiss_hits] == [
        hit.item_id for hit in sqlite_hits
    ]
    np.testing.assert_allclose(
        [hit.raw_value for hit in faiss_hits],
        [hit.raw_value for hit in sqlite_hits],
        atol=1e-6,
    )
    assert "duplicate" not in [hit.item_id for hit in faiss_hits]
    assert {"hashless-a", "hashless-b"}.issubset(
        {hit.item_id for hit in faiss_hits}
    )


def test_faiss_exact_l2_matches_sqlite_and_excludes_target(tmp_path: Path):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    build_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="l2",
    )
    faiss_search = FaissCatalogSearch(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="l2",
    )
    sqlite_search = SqliteLinearSearch(
        repository,
        model_name="clip",
        model_version="v1",
        metric="l2",
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    faiss_hits = faiss_search.search(
        query,
        ["shoes"],
        4,
        exclude_item_id="near",
    )
    sqlite_hits = sqlite_search.search(
        query,
        ["shoes"],
        4,
        exclude_item_id="near",
    )
    assert [hit.item_id for hit in faiss_hits] == [
        hit.item_id for hit in sqlite_hits
    ]
    np.testing.assert_allclose(
        [hit.raw_value for hit in faiss_hits],
        [hit.raw_value for hit in sqlite_hits],
        atol=1e-6,
    )


def test_faiss_audit_detects_stale_and_missing_indexes(tmp_path: Path):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    build_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="cosine",
    )
    ready = audit_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        expected_metric="cosine",
    )
    assert ready.ready

    (root / "clip" / "v1" / "shoes.faiss").unlink()
    stale = audit_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        expected_metric="cosine",
    )
    assert not stale.ready
    with pytest.raises(VectorIndexError):
        FaissCatalogSearch(
            repository,
            root=root,
            model_name="clip",
            model_version="v1",
            metric="cosine",
        )


def test_faiss_build_refuses_overwrite_without_replace(tmp_path: Path):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    arguments = {
        "root": root,
        "model_name": "clip",
        "model_version": "v1",
        "metric": "cosine",
    }
    build_faiss_indexes(repository, **arguments)
    with pytest.raises(FileExistsError):
        build_faiss_indexes(repository, **arguments)
    replacement = build_faiss_indexes(repository, replace=True, **arguments)
    assert replacement.categories["shoes"].count == 5


def test_faiss_audit_detects_same_count_vector_update(tmp_path: Path):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    build_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="cosine",
    )
    model = repository.get_model("clip", "v1")
    repository.upsert_embedding(
        model_id=model.id,
        item_id="far",
        category="shoes",
        image_key="shoes/far.jpg",
        image_hash="far",
        vector=np.array([0.5, 0.5], dtype=np.float32),
        force=True,
    )

    audit = audit_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        expected_metric="cosine",
    )
    assert not audit.ready
    assert any("changed after index build" in detail for detail in audit.details)


def test_faiss_audit_detects_corrupt_file_and_manifest_dimension(tmp_path: Path):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    build_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="cosine",
    )
    directory = root / "clip" / "v1"
    shoes_index = directory / "shoes.faiss"
    shoes_index.write_bytes(shoes_index.read_bytes() + b"corrupt")
    corrupt = audit_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        expected_metric="cosine",
    )
    assert not corrupt.ready
    assert "shoes: checksum mismatch" in corrupt.details

    build_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        metric="cosine",
        replace=True,
    )
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dimension"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong_dimension = audit_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        expected_metric="cosine",
    )
    assert not wrong_dimension.ready
    assert "manifest dimension does not match SQLite" in wrong_dimension.details


def test_failed_atomic_replacement_restores_previous_index(
    tmp_path: Path,
    monkeypatch,
):
    repository = _fixture(tmp_path)
    root = tmp_path / "indexes"
    arguments = {
        "root": root,
        "model_name": "clip",
        "model_version": "v1",
        "metric": "cosine",
    }
    original_manifest = build_faiss_indexes(repository, **arguments)
    real_replace = faiss_module.os.replace
    target = root / "clip" / "v1"

    def fail_install(source, destination):
        if ".build-" in str(source) and Path(destination) == target:
            raise OSError("simulated install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(faiss_module.os, "replace", fail_install)
    with pytest.raises(OSError, match="simulated install failure"):
        build_faiss_indexes(repository, replace=True, **arguments)

    restored = audit_faiss_indexes(
        repository,
        root=root,
        model_name="clip",
        model_version="v1",
        expected_metric="cosine",
    )
    assert restored.ready
    assert original_manifest.categories["shoes"].sha256 == (
        faiss_module.FaissManifest.load(target / "manifest.json")
        .categories["shoes"]
        .sha256
    )
