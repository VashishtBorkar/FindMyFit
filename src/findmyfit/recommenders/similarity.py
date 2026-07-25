"""Visual-similarity recommendations in CLIP space."""

from __future__ import annotations

import numpy as np

from findmyfit.core.models import ClothingItem, ClothingRecommendation
from findmyfit.embeddings.clip import ClipEmbedder
from findmyfit.recommenders.common import rank_catalog_candidates
from findmyfit.retrieval.sqlite_catalog import SqliteEmbeddingCatalog
from findmyfit.storage.local_images import LocalImageStore


class SimilarityRecommender:
    def __init__(
        self,
        embedder: ClipEmbedder,
        catalog: SqliteEmbeddingCatalog,
        image_store: LocalImageStore,
    ):
        self.embedder = embedder
        self.catalog = catalog
        self.image_store = image_store

    @staticmethod
    def calculate_score(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        denominator = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
        cosine = 0.0 if denominator == 0 else float(np.dot(embedding_a, embedding_b) / denominator)
        return (cosine + 1.0) / 2.0

    def recommend(
        self,
        target_item: ClothingItem,
        match_categories: list[str],
        max_recommendations: int,
    ) -> list[ClothingRecommendation]:
        target_embedding = self.embedder.embed(target_item.image_path)
        return rank_catalog_candidates(
            target_item=target_item,
            target_embedding=target_embedding,
            match_categories=match_categories,
            max_recommendations=max_recommendations,
            catalog=self.catalog,
            image_store=self.image_store,
            scorer=self.calculate_score,
        )
