"""Learned compatibility recommendations in metric space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from findmyfit.core.models import ClothingItem, ClothingRecommendation
from findmyfit.embeddings.clip import ClipEmbedder
from findmyfit.recommenders.common import recommendations_from_hits
from findmyfit.retrieval.base import VectorSearch
from findmyfit.storage.local_images import LocalImageStore

if TYPE_CHECKING:
    from findmyfit.embeddings.metric_projector import MetricProjector


class CompatibilityRecommender:
    def __init__(
        self,
        embedder: ClipEmbedder,
        projector: MetricProjector,
        search: VectorSearch,
        image_store: LocalImageStore,
    ):
        self.embedder = embedder
        self.projector = projector
        self.search = search
        self.image_store = image_store

    @staticmethod
    def score_from_raw(raw_value: float) -> float:
        distance = np.sqrt(max(raw_value, 0.0))
        return float(1.0 / (1.0 + distance))

    def recommend(
        self,
        target_item: ClothingItem,
        match_categories: list[str],
        max_recommendations: int,
    ) -> list[ClothingRecommendation]:
        clip_embedding = self.embedder.embed(target_item.image_path)
        target_embedding = self.projector.project(clip_embedding)
        hits = self.search.search(
            target_embedding,
            match_categories,
            max_recommendations,
            exclude_item_id=target_item.id,
        )
        return recommendations_from_hits(
            hits,
            image_store=self.image_store,
            score_from_raw=self.score_from_raw,
        )
