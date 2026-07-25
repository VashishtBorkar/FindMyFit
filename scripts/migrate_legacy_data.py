"""Import configured .npy embeddings and image metadata into the catalog database."""

from __future__ import annotations

import argparse
import hashlib

import numpy as np

from findmyfit.config import Settings
from findmyfit.db.models import Embedding, Image, Model
from findmyfit.db.session import create_session_factory


def compute_hash(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_or_create_model(session, name: str, version: str, dimension: int) -> int:
    model = session.query(Model).filter_by(name=name, version=version).one_or_none()
    if model is None:
        model = Model(name=name, version=version, embedding_dim=dimension)
        session.add(model)
        session.flush()
    return model.id


def main(limit: int | None = None) -> None:
    settings = Settings.from_env()
    session_factory = create_session_factory(settings.database_url)
    with session_factory.begin() as session:
        clip_model_id = get_or_create_model(
            session, "clip", settings.clip_model_version, 512
        )
        metric_model_id = get_or_create_model(
            session, settings.metric_model_name, settings.metric_model_version, 256
        )
        count = 0
        for category_dir in settings.images_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for image_path in category_dir.iterdir():
                if not image_path.is_file():
                    continue
                if limit is not None and count >= limit:
                    return
                image_id = image_path.stem
                image_key = image_path.relative_to(settings.images_dir).as_posix()
                session.merge(
                    Image(
                        id=image_id,
                        file_path=image_key,
                        category=category_dir.name,
                        hash=compute_hash(image_path),
                    )
                )
                session.flush()
                for model_id, directory, dimension in (
                    (clip_model_id, settings.clip_embeddings_dir, 512),
                    (metric_model_id, settings.metric_embeddings_dir, 256),
                ):
                    embedding_path = directory / category_dir.name / f"{image_id}.npy"
                    exists = (
                        session.query(Embedding)
                        .filter_by(image_id=image_id, model_id=model_id)
                        .one_or_none()
                    )
                    if embedding_path.is_file() and exists is None:
                        vector = np.load(embedding_path).astype(np.float32)
                        session.add(
                            Embedding(
                                image_id=image_id,
                                model_id=model_id,
                                vector=vector.tobytes(),
                                dim=dimension,
                                dtype="float32",
                            )
                        )
                count += 1
        print(f"Imported {count} catalog images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    main(arguments.limit)
