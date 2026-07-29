"""Reproducible retrieval and end-to-end API performance evaluations."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from findmyfit.config import Settings
from findmyfit.core.models import CatalogVector, VectorSearchHit
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.retrieval.faiss_index import (
    FaissCatalogSearch,
    FaissManifest,
    index_directory,
)
from findmyfit.retrieval.sqlite_search import SqliteLinearSearch
from findmyfit.storage.local_images import LocalImageStore
from findmyfit.storage.path_migration import sqlite_path_from_url

SEED = 20260729
SCORE_TOLERANCE = 1e-6


def _engine_spec(settings: Settings, engine: str) -> tuple[str, str, str]:
    if engine == "cosine":
        return (
            settings.clip_catalog_model_name,
            settings.clip_model_version,
            "cosine",
        )
    return (
        settings.metric_model_name,
        settings.metric_model_version,
        "l2",
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))


def _timing_summary(seconds: list[float]) -> dict[str, float]:
    total = sum(seconds)
    return {
        "mean_ms": statistics.fmean(seconds) * 1000 if seconds else 0.0,
        "p50_ms": _percentile(seconds, 50) * 1000,
        "p95_ms": _percentile(seconds, 95) * 1000,
        "queries_per_second": len(seconds) / total if total else 0.0,
    }


def _score(metric: str, raw_value: float) -> float:
    if metric == "cosine":
        return min(1.0, max(0.0, (raw_value + 1.0) / 2.0))
    return float(1.0 / (1.0 + math.sqrt(max(raw_value, 0.0))))


def _verify_hits(
    linear: list[VectorSearchHit],
    faiss: list[VectorSearchHit],
    *,
    metric: str,
) -> None:
    linear_ids = [hit.embedding_id for hit in linear]
    faiss_ids = [hit.embedding_id for hit in faiss]
    if linear_ids != faiss_ids:
        raise RuntimeError(
            "Retrieval parity failed: "
            f"SQLite IDs {linear_ids} != FAISS IDs {faiss_ids}"
        )
    for linear_hit, faiss_hit in zip(linear, faiss):
        linear_score = _score(metric, linear_hit.raw_value)
        faiss_score = _score(metric, faiss_hit.raw_value)
        difference = abs(linear_score - faiss_score)
        if difference > SCORE_TOLERANCE:
            raise RuntimeError(
                f"Score parity failed for embedding {linear_hit.embedding_id}: "
                f"SQLite={linear_score:.9f}, FAISS={faiss_score:.9f}, "
                f"difference={difference:.9f}"
            )


def _reservoir_queries(
    repository: SqliteEmbeddingRepository,
    *,
    model_name: str,
    model_version: str,
    per_category: int = 5,
) -> dict[str, list[CatalogVector]]:
    rng = random.Random(SEED)
    selected: dict[str, list[CatalogVector]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    for record in repository.iter_vectors(
        model_name=model_name,
        model_version=model_version,
    ):
        category = record.category
        seen[category] += 1
        candidates = selected[category]
        if len(candidates) < per_category:
            candidates.append(record)
            continue
        replacement = rng.randrange(seen[category])
        if replacement < per_category:
            candidates[replacement] = record
    return dict(sorted(selected.items()))


def _size_bucket(candidate_count: int) -> str:
    if candidate_count < 2_500:
        return "small"
    if candidate_count < 7_500:
        return "medium"
    return "large"


def _environment_metadata(
    settings: Settings,
    repository: SqliteEmbeddingRepository,
) -> dict[str, Any]:
    import faiss

    database_path = sqlite_path_from_url(settings.database_url)
    index_bytes = 0
    build_seconds = 0.0
    manifests: dict[str, Any] = {}
    for engine in ("cosine", "metric"):
        model_name, model_version, _metric = _engine_spec(settings, engine)
        directory = index_directory(
            settings.faiss_index_dir,
            model_name=model_name,
            model_version=model_version,
        )
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = FaissManifest.load(manifest_path)
        bytes_for_model = sum(
            (directory / details.filename).stat().st_size
            for details in manifest.categories.values()
        )
        index_bytes += bytes_for_model
        build_seconds += manifest.build_seconds
        manifests[f"{model_name}/{model_version}"] = {
            "vectors": sum(item.count for item in manifest.categories.values()),
            "categories": len(manifest.categories),
            "build_seconds": manifest.build_seconds,
            "disk_bytes": bytes_for_model,
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python_version": sys.version.split()[0],
        "faiss_version": faiss.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "database_bytes": database_path.stat().st_size,
        "database_models": {
            f"{name}/{version}": sum(
                repository.category_counts(
                    model_name=name,
                    model_version=version,
                ).values()
            )
            for name, version, _metric in (
                _engine_spec(settings, "cosine"),
                _engine_spec(settings, "metric"),
            )
        },
        "faiss_index_bytes": index_bytes,
        "faiss_build_seconds": build_seconds,
        "indexes": manifests,
    }


def _measure_search(
    search: SqliteLinearSearch | FaissCatalogSearch,
    query: CatalogVector,
    match_categories: list[str],
) -> tuple[list[VectorSearchHit], float]:
    started = perf_counter()
    result = search.search(
        query.vector,
        match_categories,
        5,
        exclude_item_id=query.item_id,
    )
    return result, perf_counter() - started


def run_retrieval_benchmark(settings: Settings) -> Path:
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )
    report: dict[str, Any] = {
        "kind": "retrieval",
        "environment": _environment_metadata(settings, repository),
        "models": {},
    }

    for engine in ("cosine", "metric"):
        model_name, model_version, metric = _engine_spec(settings, engine)
        queries = _reservoir_queries(
            repository,
            model_name=model_name,
            model_version=model_version,
        )
        categories = list(queries)
        category_counts = repository.category_counts(
            model_name=model_name,
            model_version=model_version,
        )

        linear = SqliteLinearSearch(
            repository,
            model_name=model_name,
            model_version=model_version,
            metric=metric,
        )
        faiss_search = FaissCatalogSearch(
            repository,
            root=settings.faiss_index_dir,
            model_name=model_name,
            model_version=model_version,
            metric=metric,
        )

        workloads: list[dict[str, Any]] = []
        for category_index, category in enumerate(categories):
            three_categories = [
                categories[(category_index + offset) % len(categories)]
                for offset in range(min(3, len(categories)))
            ]
            for query in queries[category]:
                for workload_name, match_categories in (
                    ("one_category", [category]),
                    ("three_categories", three_categories),
                ):
                    candidate_count = sum(
                        category_counts[value] for value in match_categories
                    )
                    linear.search(
                        query.vector,
                        match_categories,
                        5,
                        exclude_item_id=query.item_id,
                    )
                    faiss_search.search(
                        query.vector,
                        match_categories,
                        5,
                        exclude_item_id=query.item_id,
                    )
                    linear_hits, linear_seconds = _measure_search(
                        linear,
                        query,
                        match_categories,
                    )
                    faiss_hits, faiss_seconds = _measure_search(
                        faiss_search,
                        query,
                        match_categories,
                    )
                    _verify_hits(linear_hits, faiss_hits, metric=metric)
                    workloads.append(
                        {
                            "workload": workload_name,
                            "query_category": category,
                            "match_categories": match_categories,
                            "candidate_count": candidate_count,
                            "size_bucket": _size_bucket(candidate_count),
                            "sqlite_seconds": linear_seconds,
                            "faiss_seconds": faiss_seconds,
                        }
                    )

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for workload in workloads:
            grouped[(workload["workload"], workload["size_bucket"])].append(
                workload
            )
        summaries = []
        for (workload_name, bucket), values in sorted(grouped.items()):
            sqlite_seconds = [value["sqlite_seconds"] for value in values]
            faiss_seconds = [value["faiss_seconds"] for value in values]
            sqlite_timing = _timing_summary(sqlite_seconds)
            faiss_timing = _timing_summary(faiss_seconds)
            summaries.append(
                {
                    "workload": workload_name,
                    "size_bucket": bucket,
                    "queries": len(values),
                    "candidate_count_min": min(
                        value["candidate_count"] for value in values
                    ),
                    "candidate_count_max": max(
                        value["candidate_count"] for value in values
                    ),
                    "sqlite": sqlite_timing,
                    "faiss": faiss_timing,
                    "p50_speedup": (
                        sqlite_timing["p50_ms"] / faiss_timing["p50_ms"]
                        if faiss_timing["p50_ms"]
                        else 0.0
                    ),
                }
            )

        all_sqlite_seconds = [
            value["sqlite_seconds"] for value in workloads
        ]
        all_faiss_seconds = [value["faiss_seconds"] for value in workloads]
        overall_sqlite = _timing_summary(all_sqlite_seconds)
        overall_faiss = _timing_summary(all_faiss_seconds)
        overall_speedup = (
            overall_sqlite["p50_ms"] / overall_faiss["p50_ms"]
            if overall_faiss["p50_ms"]
            else 0.0
        )
        if overall_speedup <= 1.0:
            raise RuntimeError(
                f"FAISS p50 did not improve for {model_name}/{model_version}: "
                f"{overall_speedup:.3f}x"
            )
        report["models"][f"{model_name}/{model_version}"] = {
            "engine": engine,
            "metric": metric,
            "categories": category_counts,
            "initialization": {
                "sqlite_seconds": linear.initialization_seconds,
                "faiss_seconds": faiss_search.initialization_seconds,
            },
            "summaries": summaries,
            "overall": {
                "queries": len(workloads),
                "sqlite": overall_sqlite,
                "faiss": overall_faiss,
                "p50_speedup": overall_speedup,
            },
            "samples": workloads,
            "parity": {
                "ordering": "identical",
                "score_tolerance": SCORE_TOLERANCE,
            },
        }
        del linear, faiss_search
        gc.collect()

    output = _write_result(settings, "retrieval", report)
    _write_markdown(
        settings,
        retrieval=report,
        api=_latest_result(settings, "api"),
    )
    return output


def _image_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _api_request(
    client: Any,
    *,
    image_path: Path,
    target_category: str,
    match_categories: list[str],
) -> Any:
    files: list[tuple[str, tuple[str | None, bytes | str, str | None]]] = [
        (
            "image",
            (
                image_path.name,
                image_path.read_bytes(),
                _image_content_type(image_path),
            ),
        ),
        ("target_category", (None, target_category, None)),
        *[
            ("match_categories", (None, category, None))
            for category in match_categories
        ],
        ("max_recommendations", (None, "5", None)),
    ]
    return client.post("/recommend", files=files)


def _verify_api_payloads(sqlite_payload: dict[str, Any], faiss_payload: dict[str, Any]) -> None:
    sqlite_items = sqlite_payload["recommendations"]
    faiss_items = faiss_payload["recommendations"]
    if len(sqlite_items) != len(faiss_items):
        raise RuntimeError("API parity failed: result lengths differ")
    for sqlite_item, faiss_item in zip(sqlite_items, faiss_items):
        for field in ("item_id", "category", "image_url"):
            if sqlite_item[field] != faiss_item[field]:
                raise RuntimeError(f"API parity failed for field '{field}'")
        if abs(sqlite_item["score"] - faiss_item["score"]) > SCORE_TOLERANCE:
            raise RuntimeError("API parity failed for recommendation score")


def run_api_benchmark(settings: Settings, *, engine: str | None = None) -> Path:
    from fastapi.testclient import TestClient

    from backend.main import create_app

    selected_engine = engine or settings.recommender_engine
    repository = SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )
    model_name, model_version, _metric = _engine_spec(settings, "cosine")
    queries = _reservoir_queries(
        repository,
        model_name=model_name,
        model_version=model_version,
        per_category=1,
    )
    categories = list(queries)
    cases = []
    for index, category in enumerate(categories[:5]):
        record = queries[category][0]
        image_path = repository.image_store.resolve(record.image_key)
        matches = [
            categories[(index + offset + 1) % len(categories)]
            for offset in range(min(3, max(1, len(categories) - 1)))
        ]
        cases.append((image_path, category, matches))
    if len(cases) < 5:
        raise RuntimeError("The API benchmark requires five catalog images")

    backend_results: dict[str, Any] = {}
    payloads: dict[str, list[dict[str, Any]]] = {}
    for backend in ("sqlite", "faiss"):
        runtime_settings = Settings.from_env(
            database_url=settings.database_url,
            images_dir=settings.images_dir,
            metric_checkpoint_path=settings.metric_checkpoint_path,
            faiss_index_dir=settings.faiss_index_dir,
            recommender_engine=selected_engine,
            retrieval_backend=backend,
        )
        app = create_app(runtime_settings)
        startup_started = perf_counter()
        with TestClient(app) as client:
            startup_seconds = perf_counter() - startup_started
            readiness = client.get("/health/ready")
            if readiness.status_code != 200:
                raise RuntimeError(
                    f"{backend} API did not become ready: {readiness.text}"
                )
            measured: list[float] = []
            backend_payloads: list[dict[str, Any]] = []
            for image_path, category, matches in cases:
                warmup = _api_request(
                    client,
                    image_path=image_path,
                    target_category=category,
                    match_categories=matches,
                )
                if warmup.status_code != 200:
                    raise RuntimeError(
                        f"{backend} API warmup failed: {warmup.text}"
                    )
                selected_payload: dict[str, Any] | None = None
                for _ in range(3):
                    started = perf_counter()
                    response = _api_request(
                        client,
                        image_path=image_path,
                        target_category=category,
                        match_categories=matches,
                    )
                    measured.append(perf_counter() - started)
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"{backend} API request failed: {response.text}"
                        )
                    selected_payload = response.json()
                assert selected_payload is not None
                backend_payloads.append(selected_payload)
        backend_results[backend] = {
            "startup_seconds": startup_seconds,
            **_timing_summary(measured),
            "requests": len(measured),
        }
        payloads[backend] = backend_payloads
        gc.collect()

    for sqlite_payload, faiss_payload in zip(
        payloads["sqlite"],
        payloads["faiss"],
    ):
        _verify_api_payloads(sqlite_payload, faiss_payload)
    backend_results["p50_speedup"] = (
        backend_results["sqlite"]["p50_ms"]
        / backend_results["faiss"]["p50_ms"]
        if backend_results["faiss"]["p50_ms"]
        else 0.0
    )
    report = {
        "kind": "api",
        "engine": selected_engine,
        "environment": _environment_metadata(settings, repository),
        "cases": len(cases),
        "warmups_per_case": 1,
        "measured_requests_per_case": 3,
        "results": backend_results,
        "parity": {
            "responses": "identical",
            "score_tolerance": SCORE_TOLERANCE,
        },
    }
    output = _write_result(settings, "api", report)
    _write_markdown(
        settings,
        retrieval=_latest_result(settings, "retrieval"),
        api=report,
    )
    return output


def _benchmark_dir(settings: Settings) -> Path:
    directory = settings.project_root / "data" / "benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_result(settings: Settings, kind: str, payload: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = _benchmark_dir(settings) / f"{kind}-{timestamp}.json"
    output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return output


def _latest_result(settings: Settings, kind: str) -> dict[str, Any] | None:
    candidates = sorted(_benchmark_dir(settings).glob(f"{kind}-*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _write_markdown(
    settings: Settings,
    *,
    retrieval: dict[str, Any] | None = None,
    api: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Exact FAISS Evaluation",
        "",
        (
            "This report is generated from the real local catalog by the benchmark CLI. "
            "FAISS uses exact flat indexes, so the expected result is identical ranking "
            "with lower retrieval latency—not an approximate-recall tradeoff."
        ),
        "",
    ]
    environment = (retrieval or api or {}).get("environment", {})
    if environment:
        lines.extend(
            [
                "## Environment",
                "",
                f"- Generated: {environment.get('timestamp')}",
                f"- Python: {environment.get('python_version')}",
                f"- FAISS: {environment.get('faiss_version')}",
                f"- CPU threads: {environment.get('cpu_count')}",
                f"- Database size: {environment.get('database_bytes', 0) / (1024 ** 2):.1f} MiB",
                f"- FAISS index size: {environment.get('faiss_index_bytes', 0) / (1024 ** 2):.1f} MiB",
                f"- Combined index build time: {environment.get('faiss_build_seconds', 0):.2f} s",
                "",
            ]
        )
    if retrieval:
        lines.extend(
            [
                "## Retrieval benchmark",
                "",
                (
                    "| Model | Workload | Size | SQLite mean | SQLite p50 | "
                    "SQLite p95 | FAISS mean | FAISS p50 | FAISS p95 | Speedup |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model_name, model in retrieval["models"].items():
            overall = model.get("overall")
            if overall:
                lines.append(
                    f"| {model_name} | all | all | "
                    f"{overall['sqlite']['mean_ms']:.3f} ms | "
                    f"{overall['sqlite']['p50_ms']:.3f} ms | "
                    f"{overall['sqlite']['p95_ms']:.3f} ms | "
                    f"{overall['faiss']['mean_ms']:.3f} ms | "
                    f"{overall['faiss']['p50_ms']:.3f} ms | "
                    f"{overall['faiss']['p95_ms']:.3f} ms | "
                    f"{overall['p50_speedup']:.2f}x |"
                )
            for summary in model["summaries"]:
                lines.append(
                    f"| {model_name} | {summary['workload']} | "
                    f"{summary['size_bucket']} | "
                    f"{summary['sqlite']['mean_ms']:.3f} ms | "
                    f"{summary['sqlite']['p50_ms']:.3f} ms | "
                    f"{summary['sqlite']['p95_ms']:.3f} ms | "
                    f"{summary['faiss']['mean_ms']:.3f} ms | "
                    f"{summary['faiss']['p50_ms']:.3f} ms | "
                    f"{summary['faiss']['p95_ms']:.3f} ms | "
                    f"{summary['p50_speedup']:.2f}x |"
                )
        lines.extend(
            [
                "",
                "Initialization and throughput:",
                "",
                "| Model | SQLite init | FAISS init | SQLite QPS | FAISS QPS |",
                "|---|---:|---:|---:|---:|",
                *[
                    (
                        f"| {model_name} | "
                        f"{model['initialization']['sqlite_seconds']:.3f} s | "
                        f"{model['initialization']['faiss_seconds']:.3f} s | "
                        f"{model['overall']['sqlite']['queries_per_second']:.2f} | "
                        f"{model['overall']['faiss']['queries_per_second']:.2f} |"
                    )
                    for model_name, model in retrieval["models"].items()
                    if "overall" in model
                ],
                "",
                "Item ordering was identical and scores matched within `1e-6`.",
                "",
            ]
        )
    if api:
        sqlite = api["results"]["sqlite"]
        faiss = api["results"]["faiss"]
        lines.extend(
            [
                "## API benchmark",
                "",
                "| Backend | Startup | Mean | p50 | p95 |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| SQLite | {sqlite['startup_seconds']:.3f} s | "
                    f"{sqlite['mean_ms']:.3f} ms | {sqlite['p50_ms']:.3f} ms | "
                    f"{sqlite['p95_ms']:.3f} ms |"
                ),
                (
                    f"| FAISS | {faiss['startup_seconds']:.3f} s | "
                    f"{faiss['mean_ms']:.3f} ms | {faiss['p50_ms']:.3f} ms | "
                    f"{faiss['p95_ms']:.3f} ms |"
                ),
                "",
                (
                    "End-to-end API p50 speedup: "
                    f"**{api['results']['p50_speedup']:.2f}x**. "
                    "This includes upload validation, decoding, CLIP inference, optional "
                    "metric projection, retrieval, and serialization, so the gain can be "
                    "smaller than retrieval-only speedup."
                ),
                "",
            ]
        )
    output = settings.project_root / "docs" / "FAISS_EVALUATION.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
