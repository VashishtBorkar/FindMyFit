"""Generate portable CLIP .npy files for the configured image catalog."""

import logging

import numpy as np

from findmyfit.config import Settings
from findmyfit.embeddings.clip import ClipEmbedder


LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    if not settings.images_dir.is_dir():
        raise FileNotFoundError("Configured image directory does not exist")

    settings.clip_embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedder = ClipEmbedder(settings.clip_model_name)
    processed = 0
    skipped = 0

    for category_dir in settings.images_dir.iterdir():
        if not category_dir.is_dir():
            continue
        output_dir = settings.clip_embeddings_dir / category_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in category_dir.iterdir():
            if not image_path.is_file():
                continue
            output_path = output_dir / f"{image_path.stem}.npy"
            if output_path.exists():
                skipped += 1
                continue
            try:
                np.save(output_path, embedder.embed(image_path))
                processed += 1
            except Exception:
                LOGGER.exception("Failed to embed %s", image_path.name)

    LOGGER.info("Generated %d embeddings; skipped %d existing files", processed, skipped)


if __name__ == "__main__":
    main()
