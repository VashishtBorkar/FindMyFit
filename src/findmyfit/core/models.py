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
class CatalogVector:
    embedding_id: int
    item_id: str
    category: str
    image_key: str
    image_hash: str | None
    model_name: str
    model_version: str
    artifact_fingerprint: str
    model_dimension: int
    vector: np.ndarray


@dataclass(frozen=True)
class CatalogItemRecord:
    embedding_id: int
    item_id: str
    category: str
    image_key: str
    image_hash: str | None


@dataclass(frozen=True)
class VectorSearchHit:
    embedding_id: int
    item_id: str
    category: str
    image_key: str
    image_hash: str | None
    raw_value: float
