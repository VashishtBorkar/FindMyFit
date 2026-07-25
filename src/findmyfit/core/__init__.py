"""Domain models and category definitions."""

from findmyfit.core.categories import ALLOWED_CATEGORIES, normalize_category
from findmyfit.core.models import CatalogEmbedding, ClothingItem, ClothingRecommendation

__all__ = [
    "ALLOWED_CATEGORIES",
    "CatalogEmbedding",
    "ClothingItem",
    "ClothingRecommendation",
    "normalize_category",
]
