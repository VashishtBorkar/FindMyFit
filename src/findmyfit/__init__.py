"""Public FindMyFit runtime API.

Heavy ML modules are loaded only when their public object is requested.
"""

from typing import Any


__all__ = ["ClothingRecommender", "Settings"]


def __getattr__(name: str) -> Any:
    if name == "ClothingRecommender":
        from findmyfit.clothing_recommender import ClothingRecommender

        return ClothingRecommender
    if name == "Settings":
        from findmyfit.config import Settings

        return Settings
    raise AttributeError(name)
