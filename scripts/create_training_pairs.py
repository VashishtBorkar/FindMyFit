"""Build the existing metric-learning pair cache from configured artifacts."""

import logging

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.storage.local_images import LocalImageStore
from training.metric_learning.data import load_embeddings, load_pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )
    embeddings, _ = load_embeddings(
        repository,
        model_name=settings.clip_catalog_model_name,
        model_version=settings.clip_model_version,
    )
    pairs = load_pairs(
        embeddings,
        settings.compatibility_outfits_file,
        settings.compatibility_pairs_path,
        force_reload=True,
    )
    print(f"Saved {len(pairs)} pairs to {settings.compatibility_pairs_path}")


if __name__ == "__main__":
    main()
