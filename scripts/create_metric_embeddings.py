"""Project SQLite CLIP vectors into metric space and persist them to SQLite."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.embeddings.fingerprints import metric_artifact_fingerprint
from findmyfit.embeddings.metric_projector import MetricProjector
from findmyfit.storage.local_images import LocalImageStore

LOGGER = logging.getLogger(__name__)


def generate_metric_embeddings(
    repository: SqliteEmbeddingRepository,
    projector: MetricProjector,
    *,
    clip_model_name: str,
    clip_model_version: str,
    metric_model_id: int,
    force: bool = False,
    batch_size: int = 256,
    limit: int | None = None,
) -> dict[str, int]:
    counts = {"created": 0, "updated": 0, "skipped": 0}
    batch = []
    considered = 0

    def project_pending() -> None:
        if not batch:
            return
        projected = projector.project_batch(
            np.stack([record.vector for record in batch])
        )
        for record, vector in zip(batch, projected):
            result = repository.upsert_embedding(
                model_id=metric_model_id,
                item_id=record.item_id,
                category=record.category,
                image_key=record.image_key,
                image_hash=record.image_hash,
                vector=vector,
                force=force,
            )
            counts[result] += 1
        batch.clear()

    for record in repository.iter_vectors(
        model_name=clip_model_name,
        model_version=clip_model_version,
        batch_size=batch_size,
    ):
        if limit is not None and considered >= limit:
            break
        considered += 1
        if (
            not force
            and repository.has_embedding(
                model_id=metric_model_id,
                item_id=record.item_id,
            )
        ):
            counts["skipped"] += 1
            continue
        batch.append(record)
        if len(batch) >= batch_size:
            project_pending()
    project_pending()
    LOGGER.info("Metric generation summary: %s", counts)
    return counts


def main(
    *,
    force: bool = False,
    batch_size: int = 256,
    limit: int | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    projector = MetricProjector(settings.metric_checkpoint_path)
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )
    metric_model_id = repository.ensure_model(
        name=settings.metric_model_name,
        version=settings.metric_model_version,
        dimension=projector.output_dim,
        artifact_fingerprint=metric_artifact_fingerprint(
            settings.metric_checkpoint_path,
            model_version=settings.metric_model_version,
            clip_model_version=settings.clip_model_version,
        ),
    )
    generate_metric_embeddings(
        repository,
        projector,
        clip_model_name=settings.clip_catalog_model_name,
        clip_model_version=settings.clip_model_version,
        metric_model_id=metric_model_id,
        force=force,
        batch_size=batch_size,
        limit=limit,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    main(
        force=arguments.force,
        batch_size=arguments.batch_size,
        limit=arguments.limit,
    )
