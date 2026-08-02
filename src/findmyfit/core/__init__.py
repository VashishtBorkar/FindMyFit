"""Domain models and category definitions."""

from findmyfit.core.categories import ALLOWED_CATEGORIES, normalize_category
from findmyfit.core.models import ClothingItem, ClothingRecommendation

__all__ = [
    "ALLOWED_CATEGORIES",
    "ClothingItem",
    "ClothingRecommendation",
    "normalize_category",
]
