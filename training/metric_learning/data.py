"""Offline embedding and pair dataset loading."""

from __future__ import annotations

import itertools
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)
Pair = tuple[str, str, int]


def load_embeddings(
    embeddings_dir: str | Path,
    pickle_path: str | Path | None = None,
    force_reload: bool = False,
) -> tuple[dict, dict[str, set[str]]]:
    directory = Path(embeddings_dir)
    cache = Path(pickle_path) if pickle_path else directory / "embeddings_cache.pkl"
    if cache.exists() and not force_reload:
        with cache.open("rb") as source:
            embeddings = pickle.load(source)
        return embeddings, build_category_index(embeddings)

    embeddings = {}
    for category_dir in directory.iterdir():
        if not category_dir.is_dir():
            continue
        for embedding_file in category_dir.glob("*.npy"):
            embeddings[embedding_file.stem] = {
                "category": category_dir.name,
                "embedding": np.load(embedding_file),
            }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as destination:
        pickle.dump(embeddings, destination, protocol=pickle.HIGHEST_PROTOCOL)
    return embeddings, build_category_index(embeddings)


def build_category_index(embeddings: dict) -> dict[str, set[str]]:
    category_index: defaultdict[str, set[str]] = defaultdict(set)
    for item_id, data in embeddings.items():
        category_index[data["category"]].add(item_id)
    return dict(category_index)


def load_pairs(
    embeddings: dict,
    outfit_file: Path,
    output_pickle: Path,
    force_reload: bool = False,
) -> list[Pair]:
    if output_pickle.exists() and not force_reload:
        with output_pickle.open("rb") as source:
            return pickle.load(source)

    pairs: list[Pair] = []
    with outfit_file.open("r", encoding="utf-8") as source:
        for raw_line in source:
            parts = raw_line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            for item_a, item_b in itertools.permutations(parts[1:], 2):
                if item_a in embeddings and item_b in embeddings:
                    pairs.append((item_a, item_b, label))

    output_pickle.parent.mkdir(parents=True, exist_ok=True)
    with output_pickle.open("wb") as destination:
        pickle.dump(pairs, destination)
    LOGGER.info("Generated %d compatibility pairs", len(pairs))
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: list[Pair], embeddings: dict):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        item_a, item_b, label = self.pairs[index]
        embedding_a = torch.tensor(
            self.embeddings[item_a]["embedding"], dtype=torch.float32
        )
        embedding_b = torch.tensor(
            self.embeddings[item_b]["embedding"], dtype=torch.float32
        )
        return embedding_a, embedding_b, torch.tensor(label, dtype=torch.float32)
