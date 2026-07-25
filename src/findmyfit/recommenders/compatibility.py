"""Learned compatibility recommendations in metric space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from findmyfit.core.models import ClothingItem, ClothingRecommendation
from findmyfit.embeddings.clip import ClipEmbedder
from findmyfit.recommenders.common import rank_catalog_candidates
from findmyfit.retrieval.sqlite_catalog import SqliteEmbeddingCatalog
from findmyfit.storage.local_images import LocalImageStore

if TYPE_CHECKING:
    from findmyfit.embeddings.metric_projector import MetricProjector


class CompatibilityRecommender:
    def __init__(
        self,
        embedder: ClipEmbedder,
        projector: MetricProjector,
        catalog: SqliteEmbeddingCatalog,
        image_store: LocalImageStore,
    ):
        self.embedder = embedder
        self.projector = projector
        self.catalog = catalog
        self.image_store = image_store

    @staticmethod
    def calculate_score(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        distance = np.linalg.norm(embedding_a - embedding_b)
        return float(1.0 / (1.0 + distance))

    def recommend(
        self,
        target_item: ClothingItem,
        match_categories: list[str],
        max_recommendations: int,
    ) -> list[ClothingRecommendation]:
        clip_embedding = self.embedder.embed(target_item.image_path)
        target_embedding = self.projector.project(clip_embedding)
        return rank_catalog_candidates(
            target_item=target_item,
            target_embedding=target_embedding,
            match_categories=match_categories,
            max_recommendations=max_recommendations,
            catalog=self.catalog,
            image_store=self.image_store,
            scorer=self.calculate_score,
        )
