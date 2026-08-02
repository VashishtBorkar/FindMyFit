"""Validated, batch-oriented access to catalog vectors in SQLite."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from findmyfit.core.models import CatalogItemRecord, CatalogVector
from findmyfit.db.models import Embedding, Image, Model
from findmyfit.errors import CatalogError
from findmyfit.storage.local_images import LocalImageStore


class SqliteEmbeddingRepository:
    def __init__(
        self,
        session_factory: sessionmaker,
        image_store: LocalImageStore,
    ):
        self.session_factory = session_factory
        self.image_store = image_store

    def ensure_model(
        self,
        *,
        name: str,
        version: str,
        dimension: int,
        artifact_fingerprint: str,
    ) -> int:
        with self.session_factory.begin() as session:
            model = session.query(Model).filter_by(name=name, version=version).one_or_none()
            if model is None:
                model = Model(
                    name=name,
                    version=version,
                    embedding_dim=dimension,
                    artifact_fingerprint=artifact_fingerprint,
                )
                session.add(model)
                session.flush()
                return int(model.id)

            if model.embedding_dim != dimension:
                raise CatalogError(
                    f"Model '{name}/{version}' has dimension "
                    f"{model.embedding_dim}; expected {dimension}"
                )
            if (
                model.artifact_fingerprint
                and model.artifact_fingerprint != artifact_fingerprint
            ):
                raise CatalogError(
                    f"Model '{name}/{version}' refers to a different artifact; "
                    "use a new model version"
                )
            if not model.artifact_fingerprint:
                model.artifact_fingerprint = artifact_fingerprint
            return int(model.id)

    def get_model(self, name: str, version: str) -> Model:
        with self.session_factory() as session:
            model = session.query(Model).filter_by(name=name, version=version).one_or_none()
            if model is None:
                raise CatalogError(f"Model '{name}/{version}' is not registered")
            session.expunge(model)
            return model

    def prepare_image(
        self,
        *,
        item_id: str,
        category: str,
        image_key: str,
        image_hash: str | None,
    ) -> str:
        """Persist image metadata and invalidate every vector after content changes."""
        normalized_key = self.image_store.normalize_key(image_key)
        with self.session_factory.begin() as session:
            image = session.get(Image, item_id)
            if image is None:
                session.add(
                    Image(
                        id=item_id,
                        file_path=normalized_key,
                        category=category,
                        hash=image_hash,
                    )
                )
                return "created"

            hash_changed = image.hash != image_hash
            category_changed = image.category != category
            image.file_path = normalized_key
            image.category = category
            image.hash = image_hash
            if hash_changed:
                session.query(Embedding).filter_by(image_id=item_id).delete()
                return "changed"
            if category_changed:
                for embedding in session.query(Embedding).filter_by(image_id=item_id):
                    embedding.created_at = _utc_now()
                return "changed"
            return "unchanged"

    def has_embedding(self, *, model_id: int, item_id: str) -> bool:
        with self.session_factory() as session:
            return (
                session.query(Embedding.id)
                .filter_by(model_id=model_id, image_id=item_id)
                .first()
                is not None
            )

    def iter_vectors(
        self,
        *,
        model_name: str,
        model_version: str,
        category: str | None = None,
        batch_size: int = 1000,
    ) -> Iterator[CatalogVector]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        model = self.get_model(model_name, model_version)
        last_id = 0

        while True:
            with self.session_factory() as session:
                query = (
                    session.query(
                        Embedding.id,
                        Embedding.image_id,
                        Embedding.vector,
                        Embedding.dim,
                        Embedding.dtype,
                        Image.category,
                        Image.hash,
                        Image.file_path,
                    )
                    .join(Image, Embedding.image_id == Image.id)
                    .filter(
                        Embedding.model_id == model.id,
                        Embedding.id > last_id,
                    )
                )
                if category is not None:
                    query = query.filter(Image.category == category)
                rows = query.order_by(Embedding.id).limit(batch_size).all()

            if not rows:
                return
            for (
                embedding_id,
                item_id,
                vector_bytes,
                dimension,
                dtype,
                item_category,
                image_hash,
                file_path,
            ) in rows:
                vector = np.frombuffer(vector_bytes, dtype=np.dtype(dtype))
                if vector.size != dimension or dimension != model.embedding_dim:
                    raise CatalogError(
                        f"Embedding '{item_id}' metadata does not match model "
                        f"'{model_name}/{model_version}'"
                    )
                yield CatalogVector(
                    embedding_id=int(embedding_id),
                    item_id=item_id,
                    category=item_category,
                    image_key=self.image_store.normalize_key(file_path),
                    image_hash=image_hash,
                    model_name=model.name,
                    model_version=model.version,
                    artifact_fingerprint=model.artifact_fingerprint or "",
                    model_dimension=int(model.embedding_dim),
                    vector=vector.astype(np.float32, copy=False),
                )
                last_id = int(embedding_id)

    def get_items_by_embedding_ids(
        self,
        embedding_ids: Sequence[int],
    ) -> dict[int, CatalogItemRecord]:
        if not embedding_ids:
            return {}
        with self.session_factory() as session:
            rows = (
                session.query(
                    Embedding.id,
                    Embedding.image_id,
                    Image.category,
                    Image.file_path,
                    Image.hash,
                )
                .join(Image, Embedding.image_id == Image.id)
                .filter(Embedding.id.in_(list(embedding_ids)))
                .all()
            )
        return {
            int(embedding_id): CatalogItemRecord(
                embedding_id=int(embedding_id),
                item_id=item_id,
                category=category,
                image_key=self.image_store.normalize_key(file_path),
                image_hash=image_hash,
            )
            for embedding_id, item_id, category, file_path, image_hash in rows
        }

    def get_vectors_by_embedding_ids(
        self,
        embedding_ids: Sequence[int],
    ) -> dict[int, np.ndarray]:
        if not embedding_ids:
            return {}
        with self.session_factory() as session:
            rows = (
                session.query(
                    Embedding.id,
                    Embedding.vector,
                    Embedding.dim,
                    Embedding.dtype,
                )
                .filter(Embedding.id.in_(list(embedding_ids)))
                .all()
            )
        vectors: dict[int, np.ndarray] = {}
        for embedding_id, vector_bytes, dimension, dtype in rows:
            vector = np.frombuffer(vector_bytes, dtype=np.dtype(dtype))
            if vector.size != dimension:
                raise CatalogError(
                    f"Embedding id '{embedding_id}' has inconsistent metadata"
                )
            vectors[int(embedding_id)] = vector.astype(np.float32, copy=False)
        return vectors

    def category_counts(
        self,
        *,
        model_name: str,
        model_version: str,
    ) -> dict[str, int]:
        return {
            category: int(details["count"])
            for category, details in self.category_stats(
                model_name=model_name,
                model_version=model_version,
            ).items()
        }

    def category_stats(
        self,
        *,
        model_name: str,
        model_version: str,
    ) -> dict[str, dict[str, int | str]]:
        """Return inexpensive source-revision data for index freshness checks."""
        model = self.get_model(model_name, model_version)
        with self.session_factory() as session:
            rows = (
                session.query(
                    Image.category,
                    func.count(Embedding.id),
                    func.min(Embedding.id),
                    func.max(Embedding.id),
                    func.max(Embedding.created_at),
                )
                .join(Embedding, Embedding.image_id == Image.id)
                .filter(Embedding.model_id == model.id)
                .group_by(Image.category)
                .order_by(Image.category)
                .all()
            )
        return {
            category: {
                "count": int(count),
                "min_embedding_id": int(minimum),
                "max_embedding_id": int(maximum),
                "source_updated_at": (
                    updated_at.isoformat()
                    if isinstance(updated_at, datetime)
                    else str(updated_at)
                ),
            }
            for category, count, minimum, maximum, updated_at in rows
        }

    def upsert_embedding(
        self,
        *,
        model_id: int,
        item_id: str,
        category: str,
        image_key: str,
        image_hash: str | None,
        vector: np.ndarray,
        force: bool = False,
    ) -> str:
        normalized = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        with self.session_factory.begin() as session:
            model = session.get(Model, model_id)
            if model is None:
                raise CatalogError(f"Model id '{model_id}' is not registered")
            if normalized.size != model.embedding_dim:
                raise CatalogError(
                    f"Vector has dimension {normalized.size}; "
                    f"model expects {model.embedding_dim}"
                )

            image = session.get(Image, item_id)
            image_changed = image is not None and image.hash != image_hash
            category_changed = image is not None and image.category != category
            if image is None:
                image = Image(
                    id=item_id,
                    file_path=self.image_store.normalize_key(image_key),
                    category=category,
                    hash=image_hash,
                )
                session.add(image)
                session.flush()
            else:
                if image_changed:
                    session.query(Embedding).filter_by(image_id=item_id).delete()
                image.file_path = self.image_store.normalize_key(image_key)
                image.category = category
                image.hash = image_hash

            existing = (
                session.query(Embedding)
                .filter_by(image_id=item_id, model_id=model_id)
                .one_or_none()
            )
            if existing is not None and not force and not image_changed:
                if category_changed:
                    existing.created_at = _utc_now()
                    return "updated"
                return "skipped"
            if existing is None:
                session.add(
                    Embedding(
                        image_id=item_id,
                        model_id=model_id,
                        vector=normalized.tobytes(),
                        dim=int(normalized.size),
                        dtype="float32",
                    )
                )
                return "created"

            existing.vector = normalized.tobytes()
            existing.dim = int(normalized.size)
            existing.dtype = "float32"
            existing.created_at = _utc_now()
            return "updated"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
