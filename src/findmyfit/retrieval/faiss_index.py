"""Exact category-partitioned FAISS index build, audit, and search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from findmyfit.core.models import VectorSearchHit
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.errors import VectorIndexError
from findmyfit.retrieval.base import (
    SearchMetric,
    finalize_hits,
    normalize_vector,
)

MANIFEST_VERSION = 2


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise VectorIndexError(
            "FAISS is not installed. Install the 'retrieval' project extra."
        ) from exc
    return faiss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CategoryManifest:
    filename: str
    count: int
    min_embedding_id: int
    max_embedding_id: int
    source_updated_at: str
    sha256: str


@dataclass(frozen=True)
class FaissManifest:
    schema_version: int
    model_name: str
    model_version: str
    artifact_fingerprint: str
    dimension: int
    metric: SearchMetric
    created_at: str
    build_seconds: float
    categories: dict[str, CategoryManifest]

    @classmethod
    def load(cls, path: Path) -> FaissManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["categories"] = {
                category: CategoryManifest(**details)
                for category, details in payload["categories"].items()
            }
            return cls(**payload)
        except Exception as exc:
            raise VectorIndexError(f"Invalid FAISS manifest: {path}") from exc

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class FaissAudit:
    ready: bool
    details: tuple[str, ...]
    total_vectors: int
    total_bytes: int


def index_directory(
    root: Path,
    *,
    model_name: str,
    model_version: str,
) -> Path:
    return root / model_name / model_version


def build_faiss_indexes(
    repository: SqliteEmbeddingRepository,
    *,
    root: Path,
    model_name: str,
    model_version: str,
    metric: SearchMetric,
    replace: bool = False,
) -> FaissManifest:
    faiss = _faiss()
    model = repository.get_model(model_name, model_version)
    if not model.artifact_fingerprint:
        raise VectorIndexError(
            f"Model '{model_name}/{model_version}' has no artifact fingerprint"
        )
    target = index_directory(
        root,
        model_name=model_name,
        model_version=model_version,
    )
    if target.exists() and not replace:
        raise FileExistsError(
            f"FAISS indexes already exist at {target}; pass --replace"
        )

    started = perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.build-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    category_manifests: dict[str, CategoryManifest] = {}
    try:
        source_stats = repository.category_stats(
            model_name=model_name,
            model_version=model_version,
        )
        for category, source in source_stats.items():
            records = list(
                repository.iter_vectors(
                    model_name=model_name,
                    model_version=model_version,
                    category=category,
                )
            )
            if not records:
                continue
            source_vectors = (
                [normalize_vector(record.vector) for record in records]
                if metric == "cosine"
                else [record.vector for record in records]
            )
            vectors = np.ascontiguousarray(
                np.stack(source_vectors),
                dtype=np.float32,
            )
            if vectors.shape[1] != model.embedding_dim:
                raise VectorIndexError(
                    f"Category '{category}' has dimension {vectors.shape[1]}; "
                    f"expected {model.embedding_dim}"
                )
            if metric == "cosine":
                base_index = faiss.IndexFlatIP(model.embedding_dim)
            else:
                base_index = faiss.IndexFlatL2(model.embedding_dim)
            index = faiss.IndexIDMap2(base_index)
            ids = np.ascontiguousarray(
                np.asarray([record.embedding_id for record in records]),
                dtype=np.int64,
            )
            index.add_with_ids(vectors, ids)
            filename = f"{category}.faiss"
            output = temporary / filename
            faiss.write_index(index, str(output))
            category_manifests[category] = CategoryManifest(
                filename=filename,
                count=int(index.ntotal),
                min_embedding_id=int(ids.min()),
                max_embedding_id=int(ids.max()),
                source_updated_at=str(source["source_updated_at"]),
                sha256=_sha256(output),
            )

        manifest = FaissManifest(
            schema_version=MANIFEST_VERSION,
            model_name=model_name,
            model_version=model_version,
            artifact_fingerprint=model.artifact_fingerprint,
            dimension=int(model.embedding_dim),
            metric=metric,
            created_at=datetime.now(timezone.utc).isoformat(),
            build_seconds=perf_counter() - started,
            categories=category_manifests,
        )
        manifest.write(temporary / "manifest.json")

        backup: Path | None = None
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def audit_faiss_indexes(
    repository: SqliteEmbeddingRepository,
    *,
    root: Path,
    model_name: str,
    model_version: str,
    expected_metric: SearchMetric,
    checksums: bool = True,
) -> FaissAudit:
    faiss = _faiss()
    directory = index_directory(
        root,
        model_name=model_name,
        model_version=model_version,
    )
    details: list[str] = []
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return FaissAudit(False, ("manifest is missing",), 0, 0)
    try:
        manifest = FaissManifest.load(manifest_path)
        model = repository.get_model(model_name, model_version)
    except Exception as exc:  # noqa: BLE001 - a corrupt artifact must become audit data
        return FaissAudit(False, (f"{type(exc).__name__}: {exc}",), 0, 0)

    if manifest.schema_version != MANIFEST_VERSION:
        details.append("manifest schema version is unsupported")
    if manifest.model_name != model_name or manifest.model_version != model_version:
        details.append("manifest model identity does not match")
    if manifest.dimension != model.embedding_dim:
        details.append("manifest dimension does not match SQLite")
    if manifest.artifact_fingerprint != model.artifact_fingerprint:
        details.append("manifest artifact fingerprint is stale")
    if manifest.metric != expected_metric:
        details.append("manifest metric does not match")

    expected_stats = repository.category_stats(
        model_name=model_name,
        model_version=model_version,
    )
    if set(manifest.categories) != set(expected_stats):
        details.append("manifest categories do not match SQLite")

    total_vectors = 0
    total_bytes = 0
    for category, category_manifest in manifest.categories.items():
        path = directory / category_manifest.filename
        if not path.is_file():
            details.append(f"{category}: index file is missing")
            continue
        total_bytes += path.stat().st_size
        if checksums and _sha256(path) != category_manifest.sha256:
            details.append(f"{category}: checksum mismatch")
            continue
        try:
            index = faiss.read_index(str(path))
        except Exception:  # noqa: BLE001 - FAISS exposes backend-specific errors
            details.append(f"{category}: index cannot be loaded")
            continue
        total_vectors += int(index.ntotal)
        if index.d != manifest.dimension:
            details.append(f"{category}: dimension mismatch")
        if int(index.ntotal) != category_manifest.count:
            details.append(f"{category}: manifest count mismatch")
        source = expected_stats.get(category)
        if source is None:
            details.append(f"{category}: category is missing from SQLite")
            continue
        if int(source["count"]) != category_manifest.count:
            details.append(f"{category}: SQLite count mismatch")
        if int(source["min_embedding_id"]) != category_manifest.min_embedding_id:
            details.append(f"{category}: SQLite minimum ID mismatch")
        if int(source["max_embedding_id"]) != category_manifest.max_embedding_id:
            details.append(f"{category}: SQLite maximum ID mismatch")
        if str(source["source_updated_at"]) != category_manifest.source_updated_at:
            details.append(f"{category}: SQLite vectors changed after index build")
    return FaissAudit(not details, tuple(details), total_vectors, total_bytes)


class FaissCatalogSearch:
    def __init__(
        self,
        repository: SqliteEmbeddingRepository,
        *,
        root: Path,
        model_name: str,
        model_version: str,
        metric: SearchMetric,
    ):
        started = perf_counter()
        self.repository = repository
        self.root = root
        self.model_name = model_name
        self.model_version = model_version
        self.metric = metric
        audit = audit_faiss_indexes(
            repository,
            root=root,
            model_name=model_name,
            model_version=model_version,
            expected_metric=metric,
            checksums=False,
        )
        if not audit.ready:
            raise VectorIndexError("; ".join(audit.details))

        self.directory = index_directory(
            root,
            model_name=model_name,
            model_version=model_version,
        )
        self.manifest = FaissManifest.load(self.directory / "manifest.json")
        faiss = _faiss()
        self.indexes = {
            category: faiss.read_index(
                str(self.directory / details.filename)
            )
            for category, details in self.manifest.categories.items()
        }
        self.initialization_seconds = perf_counter() - started

    def _search_category(
        self,
        category: str,
        query: np.ndarray,
        count: int,
    ) -> tuple[list[int], list[float]]:
        index = self.indexes[category]
        distances, ids = index.search(query.reshape(1, -1), count)
        valid = ids[0] >= 0
        return (
            [int(value) for value in ids[0][valid]],
            [float(value) for value in distances[0][valid]],
        )

    def search(
        self,
        query_vector: np.ndarray,
        match_categories: list[str],
        top_k: int,
        exclude_item_id: str | None = None,
    ) -> list[VectorSearchHit]:
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(-1)
        if query.size != self.manifest.dimension:
            raise ValueError(
                f"Query has dimension {query.size}; "
                f"expected {self.manifest.dimension}"
            )
        if self.metric == "cosine":
            query = normalize_vector(query)

        categories = [
            category for category in dict.fromkeys(match_categories)
            if category in self.indexes
        ]
        fetch_counts = {
            category: min(max(top_k, 1), int(self.indexes[category].ntotal))
            for category in categories
        }
        while categories:
            raw_by_id: dict[int, float] = {}
            boundaries: dict[str, float] = {}
            for category in categories:
                ids, values = self._search_category(
                    category,
                    query,
                    fetch_counts[category],
                )
                raw_by_id.update(zip(ids, values))
                if values:
                    boundaries[category] = values[-1]
            metadata = self.repository.get_items_by_embedding_ids(list(raw_by_id))
            source_vectors = self.repository.get_vectors_by_embedding_ids(
                list(raw_by_id)
            )
            rescored: dict[int, float] = {}
            for embedding_id, vector in source_vectors.items():
                if self.metric == "cosine":
                    rescored[embedding_id] = float(
                        np.dot(query, normalize_vector(vector))
                    )
                else:
                    difference = query - vector
                    rescored[embedding_id] = float(
                        np.dot(difference, difference)
                    )
            candidates = [
                VectorSearchHit(
                    embedding_id=embedding_id,
                    item_id=item.item_id,
                    category=item.category,
                    image_key=item.image_key,
                    image_hash=item.image_hash,
                    raw_value=rescored[embedding_id],
                )
                for embedding_id, item in metadata.items()
                if embedding_id in rescored
            ]
            results = finalize_hits(
                candidates,
                metric=self.metric,
                top_k=top_k,
                exclude_item_id=exclude_item_id,
            )

            expandable = [
                category
                for category in categories
                if fetch_counts[category] < int(self.indexes[category].ntotal)
            ]
            if not expandable:
                return results
            if len(results) < top_k:
                to_expand = expandable
            else:
                threshold = results[-1].raw_value
                if self.metric == "cosine":
                    to_expand = [
                        category
                        for category in expandable
                        if boundaries.get(category, -math.inf) >= threshold - 1e-5
                    ]
                else:
                    to_expand = [
                        category
                        for category in expandable
                        if boundaries.get(category, math.inf) <= threshold + 1e-5
                    ]
            if not to_expand:
                return results
            for category in to_expand:
                fetch_counts[category] = min(
                    max(fetch_counts[category] * 2, fetch_counts[category] + 1),
                    int(self.indexes[category].ntotal),
                )
        return []
