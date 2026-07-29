from pathlib import Path

import numpy as np

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.models import Base
from findmyfit.db.session import create_engine_for_url, create_session_factory
from findmyfit.storage.local_images import LocalImageStore
from scripts.create_clip_embeddings import generate_clip_embeddings
from scripts.create_metric_embeddings import generate_metric_embeddings


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, _path: Path) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeProjector:
    def __init__(self):
        self.calls = 0

    def project_batch(self, vectors: np.ndarray) -> np.ndarray:
        self.calls += 1
        return vectors[:, :1] * 0.5


def _fixture(tmp_path: Path):
    images_dir = tmp_path / "images"
    image = images_dir / "shoes" / "shoe.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"first")
    database_url = f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}"
    Base.metadata.create_all(create_engine_for_url(database_url))
    repository = SqliteEmbeddingRepository(
        create_session_factory(database_url),
        LocalImageStore(images_dir),
    )
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        database_url=database_url,
        images_dir=images_dir,
    )
    return settings, repository, image


def test_clip_generation_resumes_and_invalidates_changed_images(tmp_path: Path):
    settings, repository, image = _fixture(tmp_path)
    clip_model_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="clip",
    )
    metric_model_id = repository.ensure_model(
        name="metric",
        version="v1",
        dimension=1,
        artifact_fingerprint="metric",
    )
    embedder = FakeEmbedder()

    first = generate_clip_embeddings(
        settings,
        repository,
        embedder,
        model_id=clip_model_id,
    )
    assert first["created"] == 1
    resumed = generate_clip_embeddings(
        settings,
        repository,
        embedder,
        model_id=clip_model_id,
    )
    assert resumed["skipped"] == 1
    assert embedder.calls == 1

    [clip_record] = list(
        repository.iter_vectors(model_name="clip", model_version="v1")
    )
    repository.upsert_embedding(
        model_id=metric_model_id,
        item_id="shoe",
        category="shoes",
        image_key="shoes/shoe.jpg",
        image_hash=clip_record.image_hash,
        vector=np.array([0.5], dtype=np.float32),
    )
    image.write_bytes(b"second")
    changed = generate_clip_embeddings(
        settings,
        repository,
        embedder,
        model_id=clip_model_id,
    )
    assert changed["created"] == 1
    assert embedder.calls == 2
    assert not repository.has_embedding(
        model_id=metric_model_id,
        item_id="shoe",
    )


def test_metric_generation_projects_only_missing_rows_in_batches(tmp_path: Path):
    _settings, repository, _image = _fixture(tmp_path)
    clip_model_id = repository.ensure_model(
        name="clip",
        version="v1",
        dimension=2,
        artifact_fingerprint="clip",
    )
    metric_model_id = repository.ensure_model(
        name="metric",
        version="v1",
        dimension=1,
        artifact_fingerprint="metric",
    )
    for index in range(3):
        repository.upsert_embedding(
            model_id=clip_model_id,
            item_id=f"item-{index}",
            category="shoes",
            image_key=f"shoes/item-{index}.jpg",
            image_hash=f"hash-{index}",
            vector=np.array([index + 1.0, 0.0], dtype=np.float32),
        )
    projector = FakeProjector()

    first = generate_metric_embeddings(
        repository,
        projector,
        clip_model_name="clip",
        clip_model_version="v1",
        metric_model_id=metric_model_id,
        batch_size=2,
    )
    assert first["created"] == 3
    assert projector.calls == 2
    resumed = generate_metric_embeddings(
        repository,
        projector,
        clip_model_name="clip",
        clip_model_version="v1",
        metric_model_id=metric_model_id,
        batch_size=2,
    )
    assert resumed["skipped"] == 3
    assert projector.calls == 2
