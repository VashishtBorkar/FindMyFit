from pathlib import Path

import numpy as np
import pytest

from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.models import Base, Embedding
from findmyfit.db.session import create_engine_for_url, create_session_factory
from findmyfit.errors import CatalogError
from findmyfit.storage.local_images import LocalImageStore
from training.metric_learning.data import load_embeddings


def _repository(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}"
    Base.metadata.create_all(create_engine_for_url(database_url))
    session_factory = create_session_factory(database_url)
    return (
        SqliteEmbeddingRepository(
            session_factory,
            LocalImageStore(tmp_path / "images"),
        ),
        session_factory,
    )


def test_repository_streams_batches_and_fetches_metadata(tmp_path: Path):
    repository, _ = _repository(tmp_path)
    model_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="clip-fingerprint",
    )
    for index in range(3):
        result = repository.upsert_embedding(
            model_id=model_id,
            item_id=f"item-{index}",
            category="shoes",
            image_key=f"shoes/item-{index}.jpg",
            image_hash=f"hash-{index}",
            vector=np.array([index, 1.0], dtype=np.float32),
        )
        assert result == "created"

    records = list(
        repository.iter_vectors(
            model_name="clip",
            model_version="v1",
            batch_size=1,
        )
    )
    assert [record.item_id for record in records] == [
        "item-0",
        "item-1",
        "item-2",
    ]
    metadata = repository.get_items_by_embedding_ids(
        [record.embedding_id for record in records]
    )
    assert metadata[records[0].embedding_id].image_key == "shoes/item-0.jpg"


def test_changed_image_hash_invalidates_all_model_embeddings(tmp_path: Path):
    repository, session_factory = _repository(tmp_path)
    clip_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="clip",
    )
    metric_id = repository.ensure_model(
        name="metric",
        version="v1",
        dimension=2,
        artifact_fingerprint="metric",
    )
    common = {
        "item_id": "item",
        "category": "shoes",
        "image_key": "shoes/item.jpg",
    }
    repository.upsert_embedding(
        model_id=clip_id,
        image_hash="old",
        vector=np.array([1.0, 0.0], dtype=np.float32),
        **common,
    )
    repository.upsert_embedding(
        model_id=metric_id,
        image_hash="old",
        vector=np.array([0.0, 1.0], dtype=np.float32),
        **common,
    )

    repository.upsert_embedding(
        model_id=clip_id,
        image_hash="new",
        vector=np.array([0.5, 0.5], dtype=np.float32),
        **common,
    )
    with session_factory() as session:
        rows = session.query(Embedding).filter_by(image_id="item").all()
    assert [(row.model_id, row.dim) for row in rows] == [(clip_id, 2)]


def test_model_versions_are_immutable(tmp_path: Path):
    repository, _ = _repository(tmp_path)
    repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="first",
    )
    with pytest.raises(CatalogError, match="different artifact"):
        repository.ensure_model(
            name="clip",
            version="v1",
            dimension=2,
            artifact_fingerprint="second",
        )


def test_upsert_rejects_wrong_vector_dimension(tmp_path: Path):
    repository, _ = _repository(tmp_path)
    model_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="clip",
    )
    with pytest.raises(CatalogError, match="model expects 2"):
        repository.upsert_embedding(
            model_id=model_id,
            item_id="item",
            category="shoes",
            image_key="shoes/item.jpg",
            image_hash="hash",
            vector=np.ones(3, dtype=np.float32),
        )


def test_training_loader_reads_clip_vectors_from_sqlite(tmp_path: Path):
    repository, _ = _repository(tmp_path)
    model_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="clip",
    )
    repository.upsert_embedding(
        model_id=model_id,
        item_id="shoe",
        category="shoes",
        image_key="shoes/shoe.jpg",
        image_hash="hash",
        vector=np.array([0.25, 0.75], dtype=np.float32),
    )

    embeddings, categories = load_embeddings(
        repository,
        model_name="clip",
        model_version="v1",
    )
    np.testing.assert_array_equal(
        embeddings["shoe"]["embedding"],
        np.array([0.25, 0.75], dtype=np.float32),
    )
    assert categories == {"shoes": {"shoe"}}
