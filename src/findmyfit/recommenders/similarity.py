"""Visual-similarity recommendations in CLIP space."""

from __future__ import annotations

from findmyfit.core.models import ClothingItem, ClothingRecommendation
from findmyfit.embeddings.clip import ClipEmbedder
from findmyfit.recommenders.common import recommendations_from_hits
from findmyfit.retrieval.base import VectorSearch
from findmyfit.storage.local_images import LocalImageStore


class SimilarityRecommender:
    def __init__(
        self,
        embedder: ClipEmbedder,
        search: VectorSearch,
        image_store: LocalImageStore,
    ):
        self.embedder = embedder
        self.search = search
        self.image_store = image_store

    @staticmethod
    def score_from_raw(raw_value: float) -> float:
        return min(1.0, max(0.0, (raw_value + 1.0) / 2.0))

    def recommend(
        self,
        target_item: ClothingItem,
        match_categories: list[str],
        max_recommendations: int,
    ) -> list[ClothingRecommendation]:
        target_embedding = self.embedder.embed(target_item.image_path)
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
