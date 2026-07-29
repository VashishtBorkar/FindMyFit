"""One-time import of legacy per-item .npy vectors into SQLite."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.embeddings.fingerprints import (
    clip_artifact_fingerprint,
    metric_artifact_fingerprint,
)
from findmyfit.storage.hashing import sha256_path
from findmyfit.storage.local_images import LocalImageStore

LOGGER = logging.getLogger(__name__)


def main(
    *,
    clip_dir: Path,
    metric_dir: Path,
    limit: int | None = None,
) -> None:
    settings = Settings.from_env()
    checkpoint = torch.load(
        settings.metric_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    metric_dimension = int(checkpoint["output_dim"])
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )
    clip_model_id = repository.ensure_model(
        name=settings.clip_catalog_model_name,
        version=settings.clip_model_version,
        dimension=512,
        artifact_fingerprint=clip_artifact_fingerprint(
            settings.clip_model_name,
            settings.clip_model_version,
        ),
    )
    metric_model_id = repository.ensure_model(
        name=settings.metric_model_name,
        version=settings.metric_model_version,
        dimension=metric_dimension,
        artifact_fingerprint=metric_artifact_fingerprint(
            settings.metric_checkpoint_path,
            model_version=settings.metric_model_version,
            clip_model_version=settings.clip_model_version,
        ),
    )

    counts = {"created": 0, "updated": 0, "skipped": 0, "missing": 0}
    visited = 0
    for category_dir in sorted(settings.images_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for image_path in sorted(category_dir.iterdir()):
            if not image_path.is_file():
                continue
            if limit is not None and visited >= limit:
                LOGGER.info("Legacy import summary: %s", counts)
                return
            visited += 1
            common = {
                "item_id": image_path.stem,
                "category": category_dir.name,
                "image_key": image_path.relative_to(settings.images_dir).as_posix(),
                "image_hash": sha256_path(image_path),
            }
            for model_id, directory, expected_dimension in (
                (clip_model_id, clip_dir, 512),
                (metric_model_id, metric_dir, metric_dimension),
            ):
                embedding_path = (
                    directory / category_dir.name / f"{image_path.stem}.npy"
                )
                if not embedding_path.is_file():
                    counts["missing"] += 1
                    continue
                vector = np.load(embedding_path).astype(np.float32).reshape(-1)
                if vector.size != expected_dimension:
                    raise ValueError(
                        f"{embedding_path} has dimension {vector.size}; "
                        f"expected {expected_dimension}"
                    )
                result = repository.upsert_embedding(
                    model_id=model_id,
                    vector=vector,
                    **common,
                )
                counts[result] += 1
    LOGGER.info("Legacy import summary: %s", counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--metric-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    main(
        clip_dir=arguments.clip_dir,
        metric_dir=arguments.metric_dir,
        limit=arguments.limit,
    )
