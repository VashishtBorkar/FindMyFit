from pathlib import Path

import numpy as np

from findmyfit.core.models import ClothingItem, VectorSearchHit
from findmyfit.recommenders.compatibility import CompatibilityRecommender
from findmyfit.recommenders.similarity import SimilarityRecommender
from findmyfit.storage.local_images import LocalImageStore


class FakeEmbedder:
    def embed(self, _path):
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeProjector:
    def project(self, vector):
        return vector


class FakeSearch:
    def __init__(self, raw_values):
        self.raw_values = raw_values

    def search(self, *_args, **_kwargs):
        return [
            VectorSearchHit(
                embedding_id=index,
                item_id=item_id,
                category="shoes",
                image_key=f"shoes/{item_id}",
                image_hash=item_id,
                raw_value=raw_value,
            )
            for index, (item_id, raw_value) in enumerate(
                self.raw_values,
                start=1,
            )
        ]


def _fixture(tmp_path: Path):
    images = tmp_path / "images"
    (images / "shoes").mkdir(parents=True)
    target = tmp_path / "target.png"
    target.write_bytes(b"target")
    return LocalImageStore(images), ClothingItem("target", target, "top")


def test_similarity_recommender_preserves_cosine_scores(tmp_path: Path):
    store, item = _fixture(tmp_path)
    recommender = SimilarityRecommender(
        FakeEmbedder(),
        FakeSearch([("near.jpg", 1.0), ("far.jpg", 0.0)]),
        store,
    )
    results = recommender.recommend(item, ["shoes"], 2)
    assert [result.recommended_item.id for result in results] == [
        "near.jpg",
        "far.jpg",
    ]
    assert [result.confidence_score for result in results] == [1.0, 0.5]


def test_compatibility_recommender_converts_squared_l2_scores(tmp_path: Path):
    store, item = _fixture(tmp_path)
    recommender = CompatibilityRecommender(
        FakeEmbedder(),
        FakeProjector(),
        FakeSearch([("near.jpg", 0.0), ("far.jpg", 2.0)]),
        store,
    )
    results = recommender.recommend(item, ["shoes"], 2)
    assert [result.recommended_item.id for result in results] == [
        "near.jpg",
        "far.jpg",
    ]
    assert results[0].confidence_score == 1.0
    assert results[1].confidence_score == 1.0 / (1.0 + np.sqrt(2.0))
