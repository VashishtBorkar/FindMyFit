"""Generate CLIP catalog vectors directly into configured SQLite storage."""

from __future__ import annotations

import argparse
import logging

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.embeddings.clip import ClipEmbedder
from findmyfit.embeddings.fingerprints import clip_artifact_fingerprint
from findmyfit.storage.hashing import sha256_path
from findmyfit.storage.local_images import LocalImageStore

LOGGER = logging.getLogger(__name__)


def generate_clip_embeddings(
    settings: Settings,
    repository: SqliteEmbeddingRepository,
    embedder: ClipEmbedder,
    *,
    model_id: int,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    if not settings.images_dir.is_dir():
        raise FileNotFoundError("Configured image directory does not exist")

    counts = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
    visited = 0

    for category_dir in sorted(settings.images_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for image_path in sorted(category_dir.iterdir()):
            if not image_path.is_file():
                continue
            if limit is not None and visited >= limit:
                LOGGER.info("CLIP generation summary: %s", counts)
                return counts
            visited += 1
            try:
                image_hash = sha256_path(image_path)
                repository.prepare_image(
                    item_id=image_path.stem,
                    category=category_dir.name,
                    image_key=image_path.relative_to(
                        settings.images_dir
                    ).as_posix(),
                    image_hash=image_hash,
                )
                if (
                    not force
                    and repository.has_embedding(
                        model_id=model_id,
                        item_id=image_path.stem,
                    )
                ):
                    counts["skipped"] += 1
                    continue
                result = repository.upsert_embedding(
                    model_id=model_id,
                    item_id=image_path.stem,
                    category=category_dir.name,
                    image_key=image_path.relative_to(settings.images_dir).as_posix(),
                    image_hash=image_hash,
                    vector=embedder.embed(image_path),
                    force=True,
                )
                counts[result] += 1
            except Exception:
                counts["failed"] += 1
                LOGGER.exception("Unable to embed %s", image_path.name)

    LOGGER.info("CLIP generation summary: %s", counts)
    return counts


def main(force: bool = False, limit: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )
    model_id = repository.ensure_model(
        name=settings.clip_catalog_model_name,
        version=settings.clip_model_version,
        dimension=512,
        artifact_fingerprint=clip_artifact_fingerprint(
            settings.clip_model_name,
            settings.clip_model_version,
        ),
    )
    generate_clip_embeddings(
        settings,
        repository,
        ClipEmbedder(settings.clip_model_name),
        model_id=model_id,
        force=force,
        limit=limit,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    main(force=arguments.force, limit=arguments.limit)
