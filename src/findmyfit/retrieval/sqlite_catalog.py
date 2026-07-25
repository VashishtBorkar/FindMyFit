"""Read model-specific catalog embeddings and metadata from SQLite/SQLAlchemy."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import numpy as np
from sqlalchemy.orm import sessionmaker

from findmyfit.core.models import CatalogEmbedding
from findmyfit.db.models import Embedding, Image, Model
from findmyfit.errors import CatalogError
from findmyfit.storage.local_images import LocalImageStore


LOGGER = logging.getLogger(__name__)


class SqliteEmbeddingCatalog:
    def __init__(
        self,
        session_factory: sessionmaker,
        image_store: LocalImageStore,
        *,
        model_name: str,
        model_version: str,
    ):
        self.session_factory = session_factory
        self.image_store = image_store
        self.model_name = model_name
        self.model_version = model_version
        self.records: dict[str, CatalogEmbedding] = {}
        self.category_index: dict[str, list[str]] = {}

    def load(self) -> None:
        with self.session_factory() as session:
            model = (
                session.query(Model)
                .filter_by(name=self.model_name, version=self.model_version)
                .one_or_none()
            )
            if model is None:
                raise CatalogError(
                    f"Embedding model '{self.model_name}/{self.model_version}' "
                    "is not registered"
                )

            rows = (
                session.query(
                    Embedding.image_id,
                    Embedding.vector,
                    Embedding.dim,
                    Embedding.dtype,
                    Image.category,
                    Image.hash,
                    Image.file_path,
                )
                .join(Image, Embedding.image_id == Image.id)
                .filter(Embedding.model_id == model.id)
                .all()
            )

        records: dict[str, CatalogEmbedding] = {}
        categories: defaultdict[str, list[str]] = defaultdict(list)
        for item_id, vector_bytes, dimension, dtype, category, image_hash, file_path in rows:
            vector = np.frombuffer(vector_bytes, dtype=np.dtype(dtype))
            if vector.size != dimension:
                raise CatalogError(
                    f"Embedding '{item_id}' has {vector.size} values; expected {dimension}"
                )
            image_key = self.image_store.normalize_key(file_path)
            records[item_id] = CatalogEmbedding(
                item_id=item_id,
                category=category,
                image_key=image_key,
                image_hash=image_hash,
                vector=vector.astype(np.float32, copy=False),
            )
            categories[category].append(item_id)

        self.records = records
        self.category_index = dict(categories)
        LOGGER.info(
            "Loaded %d %s/%s embeddings across %d categories",
            len(records),
            self.model_name,
            self.model_version,
            len(categories),
        )

    def iter_category(self, category: str) -> Iterable[CatalogEmbedding]:
        for item_id in self.category_index.get(category, ()):
            yield self.records[item_id]
