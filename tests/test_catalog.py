from pathlib import Path

import numpy as np

from findmyfit.db.models import Base, Embedding, Image, Model
from findmyfit.db.session import create_engine_for_url, create_session_factory
from findmyfit.retrieval.sqlite_catalog import SqliteEmbeddingCatalog
from findmyfit.storage.local_images import LocalImageStore


def test_sqlite_catalog_loads_vectors_and_portable_metadata(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}"
    Base.metadata.create_all(create_engine_for_url(database_url))
    session_factory = create_session_factory(database_url)
    image_path = tmp_path / "images" / "shoes" / "shoe.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with session_factory.begin() as session:
        model = Model(name="test", version="v1", embedding_dim=3)
        session.add(model)
        session.flush()
        session.add(
            Image(
                id="shoe.jpg",
                file_path=str(image_path),
                category="shoes",
                hash="hash",
            )
        )
        session.add(
            Embedding(
                image_id="shoe.jpg",
                model_id=model.id,
                vector=vector.tobytes(),
                dim=3,
                dtype="float32",
            )
        )

    catalog = SqliteEmbeddingCatalog(
        session_factory,
        LocalImageStore(tmp_path / "images"),
        model_name="test",
        model_version="v1",
    )
    catalog.load()

    [record] = list(catalog.iter_category("shoes"))
    assert record.image_key == "shoes/shoe.jpg"
    np.testing.assert_array_equal(record.vector, vector)
