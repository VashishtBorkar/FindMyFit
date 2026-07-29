"""Shared conversion from vector-search hits to recommendation results."""

from __future__ import annotations

from collections.abc import Callable

from findmyfit.core.models import (
    ClothingItem,
    ClothingRecommendation,
    VectorSearchHit,
)
from findmyfit.storage.local_images import LocalImageStore


def recommendations_from_hits(
    hits: list[VectorSearchHit],
    *,
    image_store: LocalImageStore,
    score_from_raw: Callable[[float], float],
) -> list[ClothingRecommendation]:
    return [
        ClothingRecommendation(
            recommended_item=ClothingItem(
                id=hit.item_id,
                category=hit.category,
                image_key=hit.image_key,
                image_path=image_store.resolve(hit.image_key, must_exist=False),
            ),
            confidence_score=score_from_raw(hit.raw_value),
        )
        for hit in hits
    ]
