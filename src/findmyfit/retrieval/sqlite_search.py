"""Legacy exact Python scan retained for fallback and benchmarking."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter

import numpy as np

from findmyfit.core.models import CatalogVector, VectorSearchHit
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.retrieval.base import (
    SearchMetric,
    finalize_hits,
    normalize_vector,
)


class SqliteLinearSearch:
    def __init__(
        self,
        repository: SqliteEmbeddingRepository,
        *,
        model_name: str,
        model_version: str,
        metric: SearchMetric,
    ):
        started = perf_counter()
        self.repository = repository
        self.model_name = model_name
        self.model_version = model_version
        self.metric = metric
        records: defaultdict[str, list[CatalogVector]] = defaultdict(list)
        for record in repository.iter_vectors(
            model_name=model_name,
            model_version=model_version,
        ):
            vector = record.vector
            if metric == "cosine":
                vector = normalize_vector(vector)
                record = CatalogVector(
                    embedding_id=record.embedding_id,
                    item_id=record.item_id,
                    category=record.category,
                    image_key=record.image_key,
                    image_hash=record.image_hash,
                    model_name=record.model_name,
                    model_version=record.model_version,
                    artifact_fingerprint=record.artifact_fingerprint,
                    model_dimension=record.model_dimension,
                    vector=vector,
                )
            records[record.category].append(record)
        self.records = dict(records)
        self.initialization_seconds = perf_counter() - started

    def search(
        self,
        query_vector: np.ndarray,
        match_categories: list[str],
        top_k: int,
        exclude_item_id: str | None = None,
    ) -> list[VectorSearchHit]:
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(-1)
        if self.metric == "cosine":
            query = normalize_vector(query)

        candidates: list[VectorSearchHit] = []
        for category in match_categories:
            for record in self.records.get(category, ()):
                if self.metric == "cosine":
                    raw_value = float(np.dot(query, record.vector))
                else:
                    difference = query - record.vector
                    raw_value = float(np.dot(difference, difference))
                candidates.append(
                    VectorSearchHit(
                        embedding_id=record.embedding_id,
                        item_id=record.item_id,
                        category=record.category,
                        image_key=record.image_key,
                        image_hash=record.image_hash,
                        raw_value=raw_value,
                    )
                )
        return finalize_hits(
            candidates,
            metric=self.metric,
            top_k=top_k,
            exclude_item_id=exclude_item_id,
        )
