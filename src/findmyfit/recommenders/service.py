"""Validate and orchestrate recommendation requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from findmyfit.core.categories import get_allowed_categories, normalize_category
from findmyfit.core.models import ClothingItem, ClothingRecommendation
from findmyfit.errors import InvalidRecommendationRequest


class RecommendationService:
    def __init__(self, recommender: Any, *, max_recommendations: int = 20):
        self.recommender = recommender
        self.max_recommendations = max_recommendations

    def get_recommendations(
        self,
        image_path: str | Path,
        target_category: str,
        match_categories: list[str],
        max_recommendations: int = 5,
    ) -> list[ClothingRecommendation]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path.name}")
        if not match_categories:
            raise InvalidRecommendationRequest("At least one match category is required")
        if not 1 <= max_recommendations <= self.max_recommendations:
            raise InvalidRecommendationRequest(
                f"max_recommendations must be between 1 and {self.max_recommendations}"
            )

        normalized_target = normalize_category(target_category)
        normalized_matches = [normalize_category(value) for value in match_categories]
        target = ClothingItem(
            id=path.stem,
            image_path=path,
            category=normalized_target,
        )
        return self.recommender.recommend(
            target,
            normalized_matches,
            max_recommendations,
        )

    @staticmethod
    def get_allowed_categories() -> list[str]:
        return get_allowed_categories()
