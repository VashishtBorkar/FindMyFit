"""Stable public facade for the recommendation runtime."""

from __future__ import annotations

from pathlib import Path

from findmyfit.config import Settings
from findmyfit.core.categories import get_allowed_categories, normalize_category
from findmyfit.core.models import ClothingRecommendation
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.embeddings.clip import ClipEmbedder
from findmyfit.errors import ConfigurationError
from findmyfit.recommenders.compatibility import CompatibilityRecommender
from findmyfit.recommenders.service import RecommendationService
from findmyfit.recommenders.similarity import SimilarityRecommender
from findmyfit.retrieval.faiss_index import FaissCatalogSearch
from findmyfit.retrieval.sqlite_search import SqliteLinearSearch
from findmyfit.storage.local_images import LocalImageStore


class ClothingRecommender:
    """Compatibility facade that constructs the concrete recommendation pipeline."""

    def __init__(
        self,
        recommendation_engine_type: str,
        images_dir: str | Path | None = None,
        *,
        settings: Settings | None = None,
    ):
        self.settings = settings or Settings.from_env()
        engine_type = recommendation_engine_type.strip().lower()
        if engine_type not in {"cosine", "metric"}:
            raise ConfigurationError(
                "recommendation_engine_type must be either 'cosine' or 'metric'"
            )

        image_store = LocalImageStore(images_dir or self.settings.images_dir)
        session_factory = create_session_factory(self.settings.database_url)
        repository = SqliteEmbeddingRepository(session_factory, image_store)
        embedder = ClipEmbedder(self.settings.clip_model_name)

        if engine_type == "cosine":
            model_name = self.settings.clip_catalog_model_name
            model_version = self.settings.clip_model_version
            metric = "cosine"
        else:
            model_name = self.settings.metric_model_name
            model_version = self.settings.metric_model_version
            metric = "l2"

        if self.settings.retrieval_backend == "faiss":
            search = FaissCatalogSearch(
                repository,
                root=self.settings.faiss_index_dir,
                model_name=model_name,
                model_version=model_version,
                metric=metric,
            )
        else:
            search = SqliteLinearSearch(
                repository,
                model_name=model_name,
                model_version=model_version,
                metric=metric,
            )

        if engine_type == "cosine":
            strategy = SimilarityRecommender(embedder, search, image_store)
        else:
            from findmyfit.embeddings.metric_projector import MetricProjector

            projector = MetricProjector(self.settings.metric_checkpoint_path)
            strategy = CompatibilityRecommender(
                embedder, projector, search, image_store
            )

        self.service = RecommendationService(
            strategy,
            max_recommendations=self.settings.max_recommendations,
        )

    def get_recommendations(
        self,
        image_path: str | Path,
        target_category: str,
        match_categories: list[str],
        max_recommendations: int = 5,
    ) -> list[ClothingRecommendation]:
        return self.service.get_recommendations(
            image_path,
            target_category,
            match_categories,
            max_recommendations,
        )

    @classmethod
    def validate_category(cls, category: str) -> str:
        return normalize_category(category)

    @classmethod
    def get_allowed_categories(cls) -> list[str]:
        return get_allowed_categories()
