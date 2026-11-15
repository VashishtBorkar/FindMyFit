"""Prepare reproducible retrieval examples from official Polyvore outfit splits."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RankingDataset(Dataset):
    def __init__(self, examples, embeddings, category_ids):
        self.examples = examples
        self.embeddings = embeddings
        self.category_ids = category_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        anchor, category, candidates, positive_index = self.examples[index]
        return (
            torch.from_numpy(self.embeddings[anchor]["embedding"]),
            self.category_ids[category],
            torch.from_numpy(np.stack([
                self.embeddings[item]["embedding"] for item in candidates
            ])),
            positive_index,
        )


def _identity(item, embeddings):
    image_hash = embeddings[item].get("image_hash")
    return f"hash:{image_hash}" if image_hash else f"id:{item}"


def _overlap(partitions):
    return {
        f"{a}/{b}": len(partitions[a] & partitions[b])
        for a, b in itertools.combinations(partitions, 2)
    }


def make_ranking_splits(
    outfit_file: Path,
    embeddings: dict,
    *,
    seed: int,
    candidate_count: int,
    anchor_category: str | None = None,
    target_categories: list[str] | None = None,
    sampling_context: dict | None = None,
) -> tuple[dict, dict]:
    """Use official JSON partitions when passed their directory.

    Legacy text input retains a seeded outfit split for small experiments.
    Exact image copies are reserved for test, then validation, then training.
    """
    if candidate_count < 2:
        raise ValueError("At least two candidates are required")
    rng = random.Random(seed)
    partitions = {}
    sources = {}
    if outfit_file.is_dir():
        for split, filename in (
            ("train", "train_no_dup.json"),
            ("validation", "valid_no_dup.json"),
            ("test", "test_no_dup.json"),
        ):
            content = (outfit_file / filename).read_bytes()
            records = json.loads(content)
            partitions[split] = [
                tuple(sorted({f"{row['set_id']}_{item['index']}" for item in row["items"]}))
                for row in records
            ]
            sources[split] = {
                "filename": filename, "sha256": hashlib.sha256(content).hexdigest()
            }
    else:
        rows = set()
        for line in outfit_file.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "1":
                rows.add(tuple(sorted(set(fields[1:]))))
        rows = sorted(rows)
        rng.shuffle(rows)
        holdout = max(1, len(rows) // 10)
        partitions = {
            "train": rows[2 * holdout:],
            "validation": rows[:holdout],
            "test": rows[holdout:2 * holdout],
        }

    audit = {"sources": sources, "coverage": {}}
    available = {}
    for split, rows in partitions.items():
        requested = set(itertools.chain.from_iterable(rows))
        found = requested.intersection(embeddings)
        available[split] = found
        audit["coverage"][split] = {
            "outfits": len(rows),
            "requested_items": len(requested),
            "available_items": len(found),
            "missing_items": len(requested - found),
            "complete_outfits": sum(all(item in embeddings for item in row) for row in rows),
        }
    identities = {
        split: {_identity(item, embeddings) for item in items}
        for split, items in available.items()
    }
    audit["original_item_overlap"] = _overlap(available)
    audit["original_image_overlap"] = _overlap(identities)
    image_pairs = {}
    for split, rows in partitions.items():
        image_pairs[split] = set()
        for row in rows:
            images = sorted({_identity(item, embeddings) for item in row if item in embeddings})
            image_pairs[split].update(itertools.combinations(images, 2))
    audit["original_image_pair_overlap"] = _overlap(image_pairs)

    reserved = set()
    canonical = {}
    removed = {}
    for split in ("test", "validation", "train"):
        representatives = {}
        for item in sorted(available[split]):
            identity = _identity(item, embeddings)
            if identity not in reserved:
                representatives.setdefault(identity, item)
        canonical[split] = {
            item: representatives[_identity(item, embeddings)]
            for item in available[split]
            if _identity(item, embeddings) in representatives
        }
        removed[split] = len(available[split]) - len(canonical[split])
        reserved.update(representatives)
    audit["items_removed_for_heldout_overlap"] = removed
    retained = {
        split: {_identity(item, embeddings) for item in mapping.values()}
        for split, mapping in canonical.items()
    }
    audit["retained_image_overlap"] = _overlap(retained)

    examples = {}
    audit["examples"] = {}
    audit["relationships"] = {}
    audit["insufficient_candidates"] = {}
    for split, rows in partitions.items():
        mapping = canonical[split]
        pool = defaultdict(list)
        for item in sorted(set(mapping.values())):
            pool[embeddings[item]["category"]].append(item)
        partners = defaultdict(set)
        pairs = set()
        for row in rows:
            items = sorted({mapping[item] for item in row if item in mapping})
            for a, b in itertools.combinations(items, 2):
                partners[a].add(b)
                partners[b].add(a)
                pairs.update(((a, b), (b, a)))
        if sampling_context is not None:
            sampling_context[split] = {"pools": dict(pool), "partners": dict(partners)}
        result = []
        counts = Counter()
        skipped = 0
        for anchor, positive in sorted(pairs):
            source_category = embeddings[anchor]["category"]
            category = embeddings[positive]["category"]
            if anchor_category is not None and source_category != anchor_category:
                continue
            if source_category == category or (target_categories and category not in target_categories):
                continue
            excluded = partners[anchor] | {anchor}
            eligible = pool[category]
            if len(eligible) - len(excluded.intersection(eligible)) < candidate_count - 1:
                skipped += 1
                continue
            negatives = []
            seen = set()
            for _ in range(4 * candidate_count):
                item = rng.choice(eligible)
                if item not in excluded and item not in seen:
                    negatives.append(item)
                    seen.add(item)
                    if len(negatives) == candidate_count - 1:
                        break
            if len(negatives) < candidate_count - 1:
                remaining = [item for item in eligible if item not in excluded | seen]
                negatives.extend(rng.sample(remaining, candidate_count - 1 - len(negatives)))
            positive_index = rng.randrange(candidate_count)
            negatives.insert(positive_index, positive)
            result.append((anchor, category, negatives, positive_index))
            counts[f"{source_category}->{category}"] += 1
        examples[split] = result
        audit["examples"][split] = len(result)
        audit["relationships"][split] = dict(counts)
        audit["insufficient_candidates"][split] = skipped
    return examples, audit


class ResampledRankingDataset(RankingDataset):
    """Deterministically refresh training alternatives without changing positives."""

    def __init__(self, examples, embeddings, category_ids, context, seed):
        super().__init__(examples, embeddings, category_ids)
        self.original_examples = examples
        self.context = context
        self.seed = seed

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + 1000003 * epoch)
        examples = []
        for anchor, category, original, index in self.original_examples:
            positive = original[index]
            excluded = self.context["partners"][anchor] | {anchor}
            pool = self.context["pools"][category]
            selected = []
            seen = set()
            while len(selected) < len(original) - 1:
                item = rng.choice(pool)
                if item not in excluded and item not in seen:
                    selected.append(item)
                    seen.add(item)
            selected.insert(index, positive)
            examples.append((anchor, category, selected, index))
        self.examples = examples
