from pathlib import Path

import numpy as np

from findmyfit.core.models import CatalogEmbedding, ClothingItem
from findmyfit.recommenders.compatibility import CompatibilityRecommender
from findmyfit.recommenders.similarity import SimilarityRecommender
from findmyfit.storage.local_images import LocalImageStore


class FakeEmbedder:
    def embed(self, _path):
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeProjector:
    def project(self, vector):
        return vector


class FakeCatalog:
    def __init__(self, records):
        self.records = records

    def iter_category(self, category):
        return (record for record in self.records if record.category == category)


def _fixture(tmp_path: Path):
    images = tmp_path / "images"
    (images / "shoes").mkdir(parents=True)
    target = tmp_path / "target.png"
    target.write_bytes(b"target")
    records = [
        CatalogEmbedding(
            "near.jpg",
            "shoes",
            "shoes/near.jpg",
            "near",
            np.array([1.0, 0.0], dtype=np.float32),
        ),
        CatalogEmbedding(
            "far.jpg",
            "shoes",
            "shoes/far.jpg",
            "far",
            np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]
    item = ClothingItem("target", target, "top")
    return LocalImageStore(images), FakeCatalog(records), item


def test_similarity_recommender_preserves_cosine_order(tmp_path: Path):
    store, catalog, item = _fixture(tmp_path)
    recommender = SimilarityRecommender(FakeEmbedder(), catalog, store)
    results = recommender.recommend(item, ["shoes"], 2)
    assert [result.recommended_item.id for result in results] == ["near.jpg", "far.jpg"]
    assert [result.confidence_score for result in results] == [1.0, 0.5]


def test_compatibility_recommender_preserves_distance_order(tmp_path: Path):
    store, catalog, item = _fixture(tmp_path)
    recommender = CompatibilityRecommender(
        FakeEmbedder(), FakeProjector(), catalog, store
    )
    results = recommender.recommend(item, ["shoes"], 2)
    assert [result.recommended_item.id for result in results] == ["near.jpg", "far.jpg"]
    assert results[0].confidence_score == 1.0
