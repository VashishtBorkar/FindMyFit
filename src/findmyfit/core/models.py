"""Small domain objects shared across the recommendation pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClothingItem:
    id: str
    image_path: Path
    category: str
    image_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_path", Path(self.image_path))


@dataclass(frozen=True)
class ClothingRecommendation:
    recommended_item: ClothingItem
    confidence_score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Recommendation score must be between 0 and 1")


@dataclass(frozen=True)
class CatalogEmbedding:
    item_id: str
    category: str
    image_key: str
    image_hash: str | None
    vector: np.ndarray
