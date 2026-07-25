"""Shared candidate ranking loop used by both active recommendation strategies."""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable

import numpy as np

from findmyfit.core.models import ClothingItem, ClothingRecommendation
from findmyfit.retrieval.sqlite_catalog import SqliteEmbeddingCatalog
from findmyfit.storage.local_images import LocalImageStore


def rank_catalog_candidates(
    *,
    target_item: ClothingItem,
    target_embedding: np.ndarray,
    match_categories: list[str],
    max_recommendations: int,
    catalog: SqliteEmbeddingCatalog,
    image_store: LocalImageStore,
    scorer: Callable[[np.ndarray, np.ndarray], float],
) -> list[ClothingRecommendation]:
    scored_items: list[tuple[float, int, ClothingItem]] = []
    seen_hashes: set[str | None] = set()
    counter = itertools.count()

    for category in match_categories:
        for record in catalog.iter_category(category):
            if target_item.id and record.item_id == target_item.id:
                continue
            if record.image_hash in seen_hashes:
                continue
            seen_hashes.add(record.image_hash)

            item = ClothingItem(
                id=record.item_id,
                category=record.category,
                image_key=record.image_key,
                image_path=image_store.resolve(record.image_key, must_exist=False),
            )
            score = scorer(target_embedding, record.vector)
            scored_items.append((score, next(counter), item))

    top_items = heapq.nlargest(max_recommendations, scored_items)
    return [
        ClothingRecommendation(recommended_item=item, confidence_score=score)
        for score, _, item in top_items
    ]
