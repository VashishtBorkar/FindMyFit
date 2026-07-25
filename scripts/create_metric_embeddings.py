"""Project configured CLIP embeddings into the trained metric space."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from findmyfit.config import Settings
from findmyfit.models.metric import FashionCompatibilityModel


LOGGER = logging.getLogger(__name__)


def load_trained_model(checkpoint_path: Path, embedding_dim: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FashionCompatibilityModel(
        embedding_dim=embedding_dim,
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def generate_metric_embeddings(
    clip_embeddings_dir: Path,
    metric_embeddings_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int = 64,
    force_reload: bool = False,
) -> tuple[int, int]:
    metric_embeddings_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for category_dir in clip_embeddings_dir.iterdir():
        if not category_dir.is_dir():
            continue
        output_dir = metric_embeddings_dir / category_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        embedding_files = list(category_dir.glob("*.npy"))
        LOGGER.info("Processing %s (%d items)", category_dir.name, len(embedding_files))

        for start in tqdm(
            range(0, len(embedding_files), batch_size),
            desc=f"Processing {category_dir.name}",
        ):
            input_vectors = []
            output_paths = []
            for source in embedding_files[start : start + batch_size]:
                destination = output_dir / source.name
                if destination.exists() and not force_reload:
                    skipped += 1
                    continue
                try:
                    input_vectors.append(np.load(source))
                    output_paths.append(destination)
                except Exception:
                    LOGGER.exception("Unable to load %s", source.name)

            if not input_vectors:
                continue
            with torch.no_grad():
                inputs = torch.as_tensor(
                    np.asarray(input_vectors), dtype=torch.float32, device=device
                )
                output_vectors = model(inputs).cpu().numpy()
            for vector, destination in zip(output_vectors, output_paths):
                np.save(destination, vector)
                processed += 1
    return processed, skipped


def main(force_reload: bool = False) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    if not settings.clip_embeddings_dir.is_dir():
        raise FileNotFoundError("Configured CLIP embedding directory does not exist")
    if not settings.metric_checkpoint_path.is_file():
        raise FileNotFoundError("Configured metric checkpoint does not exist")

    model, device = load_trained_model(settings.metric_checkpoint_path)
    processed, skipped = generate_metric_embeddings(
        settings.clip_embeddings_dir,
        settings.metric_embeddings_dir,
        model,
        device,
        force_reload=force_reload,
    )
    LOGGER.info("Generated %d metric embeddings; skipped %d", processed, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(arguments.force)
