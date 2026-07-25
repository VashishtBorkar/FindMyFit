"""Build the existing metric-learning pair cache from configured artifacts."""

import logging

from findmyfit.config import Settings
from training.metric_learning.data import load_embeddings, load_pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    embeddings, _ = load_embeddings(settings.clip_embeddings_dir)
    pairs = load_pairs(
        embeddings,
        settings.compatibility_outfits_file,
        settings.compatibility_pairs_path,
        force_reload=True,
    )
    print(f"Saved {len(pairs)} pairs to {settings.compatibility_pairs_path}")


if __name__ == "__main__":
    main()
