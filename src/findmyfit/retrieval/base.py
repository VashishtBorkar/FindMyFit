"""Shared vector-search contract and deterministic result processing."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from findmyfit.core.models import VectorSearchHit

SearchMetric = Literal["cosine", "l2"]


class VectorSearch(Protocol):
    metric: SearchMetric

    def search(
        self,
        query_vector: np.ndarray,
        match_categories: list[str],
        top_k: int,
        exclude_item_id: str | None = None,
    ) -> list[VectorSearchHit]: ...


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    normalized = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(normalized))
    if norm == 0:
        raise ValueError("Cannot search with a zero-length vector")
    return normalized / norm


def finalize_hits(
    candidates: list[VectorSearchHit],
    *,
    metric: SearchMetric,
    top_k: int,
    exclude_item_id: str | None,
) -> list[VectorSearchHit]:
    if metric == "cosine":
        ordered = sorted(
            candidates,
            key=lambda hit: (-hit.raw_value, hit.embedding_id),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda hit: (hit.raw_value, hit.embedding_id),
        )

    results: list[VectorSearchHit] = []
    seen_hashes: set[str] = set()
    for hit in ordered:
        if exclude_item_id and hit.item_id == exclude_item_id:
            continue
        if hit.image_hash is not None:
            if hit.image_hash in seen_hashes:
                continue
            seen_hashes.add(hit.image_hash)
        results.append(hit)
        if len(results) >= top_k:
            break
    return results
