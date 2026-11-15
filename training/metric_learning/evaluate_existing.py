"""Evaluate stored compatibility vectors on an experiment's frozen candidates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.storage.local_images import LocalImageStore
from training.metric_learning.compare_retrieval import evaluate, write_json
from training.metric_learning.data import load_embeddings
from training.metric_learning.ranking_data import RankingDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings.from_env()
    baseline = json.loads((args.run / "metrics.json").read_text())
    candidate_bytes = (args.run / "candidates.json").read_bytes()
    candidates_digest = hashlib.sha256(candidate_bytes).hexdigest()
    if candidates_digest != baseline["candidates_sha256"]:
        parser.error("Candidate file differs from the baseline's saved digest")
    examples = json.loads(candidate_bytes)["test"]
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url), LocalImageStore(settings.images_dir)
    )
    embeddings, _ = load_embeddings(
        repository, model_name=settings.metric_model_name,
        model_version=settings.metric_model_version,
    )
    model = repository.get_model(settings.metric_model_name, settings.metric_model_version)
    required = {item for anchor, _, candidates, _ in examples for item in [anchor, *candidates]}
    missing = required - embeddings.keys()
    if missing:
        parser.error(f"Existing model lacks vectors for {len(missing)} evaluation items")
    for item in required:
        if not np.isfinite(embeddings[item]["embedding"]).all():
            parser.error(f"Nonfinite existing vector for {item}")
    for _, category, candidates, _ in examples:
        if any(embeddings[item]["category"] != category for item in candidates):
            parser.error("Candidate categories differ from the saved experiment")
    digest = hashlib.sha256()
    for item in sorted(required):
        digest.update(item.encode() + b"\0")
        digest.update(embeddings[item]["embedding"].tobytes())
    torch.set_num_threads(4)
    dataset = RankingDataset(examples, embeddings, baseline["category_ids"])
    results = {
        "model_name": model.name,
        "model_version": model.version,
        "embedding_dim": model.embedding_dim,
        "artifact_fingerprint": model.artifact_fingerprint,
        "candidates_sha256": candidates_digest,
        "evaluation_vectors_sha256": digest.hexdigest(),
        "scoring": "negative squared Euclidean distance",
        "required_items": len(required),
        "test": evaluate(None, DataLoader(dataset, batch_size=64), torch.device("cpu"), metric="l2"),
        "training_provenance": "Previous training examples and split assignments are unknown.",
    }
    write_json(args.run / "existing_model.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
