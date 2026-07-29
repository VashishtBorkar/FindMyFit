"""Developer diagnostics and explicit path migration commands."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from findmyfit.config import Settings
from findmyfit.db.catalog_migration import audit_catalog, migrate_catalog
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.models import Model
from findmyfit.db.session import create_session_factory
from findmyfit.retrieval.faiss_index import (
    audit_faiss_indexes,
    build_faiss_indexes,
)
from findmyfit.storage.local_images import LocalImageStore
from findmyfit.storage.path_migration import (
    audit_catalog_paths,
    migrate_legacy_artifacts,
    normalize_catalog_paths,
    sqlite_path_from_url,
)

LOGGER = logging.getLogger(__name__)


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


def _repository(settings: Settings) -> SqliteEmbeddingRepository:
    return SqliteEmbeddingRepository(
        create_session_factory(settings.database_url),
        LocalImageStore(settings.images_dir),
    )


def _status(label: str, ready: bool, detail: str) -> bool:
    print(f"[{'OK' if ready else 'ERROR'}] {label}: {detail}")
    return ready


def doctor(settings: Settings) -> int:
    checks: list[bool] = []
    checks.append(
        _status(
            "images",
            settings.images_dir.is_dir(),
            str(settings.images_dir) if settings.images_dir.is_dir() else "directory is missing",
        )
    )
    if settings.recommender_engine == "cosine":
        checks.append(_status("checkpoint", True, "not required for cosine"))
    elif not settings.metric_checkpoint_path.is_file():
        checks.append(_status("checkpoint", False, "file is missing"))
    else:
        try:
            import torch

            checkpoint = torch.load(
                settings.metric_checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            required_keys = {"model_state_dict", "hidden_dim", "output_dim"}
            missing_keys = sorted(required_keys.difference(checkpoint))
            checks.append(
                _status(
                    "checkpoint",
                    not missing_keys,
                    "metadata is valid"
                    if not missing_keys
                    else f"missing keys: {', '.join(missing_keys)}",
                )
            )
        except ImportError:
            checks.append(
                _status(
                    "checkpoint",
                    True,
                    "file exists; install the ml extra to inspect metadata",
                )
            )
        except Exception as exc:
            LOGGER.debug("Checkpoint doctor check failed", exc_info=True)
            checks.append(_status("checkpoint", False, type(exc).__name__))

    try:
        catalog_audit = audit_catalog(settings)
        catalog_detail = "; ".join(
            (
                f"{model.name}/{model.version}: rows={model.row_count}, "
                f"missing={model.missing_items}, dim={model.recorded_dimension}"
            )
            for model in catalog_audit.models
        )
        checks.append(
            _status(
                "catalog",
                catalog_audit.ready,
                catalog_detail
                if catalog_audit.ready
                else "run 'python -m findmyfit catalog migrate --apply'",
            )
        )
        if not catalog_audit.ready:
            return 1

        database_path = sqlite_path_from_url(settings.database_url)
        if not database_path.is_file():
            raise FileNotFoundError("configured SQLite database is missing")
        session_factory = create_session_factory(settings.database_url)
        with session_factory() as session:
            session.execute(select(1))
            models = session.query(Model).all()
        model_names = ", ".join(f"{model.name}/{model.version}" for model in models)
        checks.append(_status("database", True, model_names or "connected; no models registered"))
        if settings.retrieval_backend == "faiss":
            name, version, metric = _engine_spec(
                settings,
                settings.recommender_engine,
            )
            index_audit = audit_faiss_indexes(
                _repository(settings),
                root=settings.faiss_index_dir,
                model_name=name,
                model_version=version,
                expected_metric=metric,
            )
            checks.append(
                _status(
                    "vector_index",
                    index_audit.ready,
                    (
                        f"{index_audit.total_vectors} vectors"
                        if index_audit.ready
                        else "; ".join(index_audit.details)
                    ),
                )
            )
        else:
            checks.append(_status("vector_index", True, "SQLite backend selected"))
    except Exception as exc:
        LOGGER.debug("Database doctor check failed", exc_info=True)
        checks.append(_status("database", False, type(exc).__name__))

    return 0 if all(checks) else 1


def catalog_command(settings: Settings, action: str, apply: bool) -> int:
    if action == "audit":
        audit = audit_catalog(settings)
        print(f"Database: {audit.database_path}")
        print(
            "Schema: "
            f"fingerprint_column={audit.fingerprint_column}; "
            f"missing_indexes={','.join(audit.missing_indexes) or 'none'}"
        )
        for model in audit.models:
            print(
                f"Model {model.name}/{model.version}: "
                f"rows={model.row_count}; missing_items={model.missing_items}; "
                f"recorded_dim={model.recorded_dimension}; "
                f"declared_dims={list(model.declared_dimensions)}; "
                f"actual_dims={list(model.actual_dimensions)}; "
                f"expected_dim={model.expected_dimension}; "
                f"fingerprint={'ok' if model.fingerprint_matches else 'missing/mismatch'}"
            )
        for error in audit.errors:
            print(f"ERROR: {error}")
        return 0 if audit.ready else 1

    audit, backup = migrate_catalog(settings, apply=apply)
    if not apply:
        print("Dry run only. Pass --apply to update catalog metadata and indexes.")
        print(
            "Required changes: "
            f"fingerprint_column={not audit.fingerprint_column}; "
            f"indexes={','.join(audit.missing_indexes) or 'none'}"
        )
        return 0 if not audit.errors else 1

    print(f"Catalog backup: {backup}")
    print(f"Catalog ready: {audit.ready}")
    return 0 if audit.ready else 1


def paths_command(settings: Settings, action: str, apply: bool) -> int:
    database_path = sqlite_path_from_url(settings.database_url)
    if not database_path.is_file():
        print("[ERROR] database: configured SQLite database is missing")
        return 1
    session_factory = create_session_factory(settings.database_url)
    image_store = LocalImageStore(settings.images_dir)
    if action == "audit":
        audit = audit_catalog_paths(session_factory, image_store)
        print(
            "Catalog rows: "
            f"{audit.total_rows}; absolute={audit.absolute_rows}; "
            f"normalized={audit.normalized_rows}; missing={audit.missing_rows}; "
            f"outside_root={audit.outside_root_rows}"
        )
        for detail in audit.details[:20]:
            print(f"  - {detail}")
        artifact_messages = migrate_legacy_artifacts(settings, apply=False)
        for message in artifact_messages:
            print(message)
        return 1 if audit.missing_rows or audit.outside_root_rows else 0

    audit, backup, updated = normalize_catalog_paths(
        session_factory,
        image_store,
        settings.database_url,
        apply=apply,
    )
    print(f"Catalog rows checked: {audit.total_rows}")
    if not apply:
        print("Dry run only. Pass --apply to rewrite verified rows and move artifacts.")
    else:
        print(f"Normalized database rows: {updated}")
        print(f"Database backup: {backup}")
    for message in migrate_legacy_artifacts(settings, apply=apply):
        print(message)
    return 0


def faiss_command(
    settings: Settings,
    action: str,
    engine: str,
    *,
    replace: bool,
) -> int:
    repository = _repository(settings)
    engines = ("cosine", "metric") if engine == "all" else (engine,)
    ready = True
    for selected_engine in engines:
        model_name, model_version, metric = _engine_spec(settings, selected_engine)
        if action == "build":
            manifest = build_faiss_indexes(
                repository,
                root=settings.faiss_index_dir,
                model_name=model_name,
                model_version=model_version,
                metric=metric,
                replace=replace,
            )
            print(
                f"Built {model_name}/{model_version}: "
                f"{sum(category.count for category in manifest.categories.values())} "
                f"vectors in {len(manifest.categories)} categories "
                f"({manifest.build_seconds:.3f}s)"
            )
            continue

        audit = audit_faiss_indexes(
            repository,
            root=settings.faiss_index_dir,
            model_name=model_name,
            model_version=model_version,
            expected_metric=metric,
        )
        ready = ready and audit.ready
        print(
            f"[{'OK' if audit.ready else 'ERROR'}] "
            f"{model_name}/{model_version}: "
            f"vectors={audit.total_vectors}; bytes={audit.total_bytes}"
        )
        for detail in audit.details:
            print(f"  - {detail}")
    return 0 if ready else 1


def benchmark_command(settings: Settings, target: str, engine: str | None) -> int:
    from findmyfit.benchmarks import (
        run_api_benchmark,
        run_retrieval_benchmark,
    )

    if target == "retrieval":
        output = run_retrieval_benchmark(settings)
    else:
        output = run_api_benchmark(settings, engine=engine)
    print(f"Benchmark result: {output}")
    print(f"Evaluation report: {settings.project_root / 'docs/FAISS_EVALUATION.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="findmyfit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check local runtime readiness")

    catalog = subparsers.add_parser("catalog", help="Audit or migrate vector metadata")
    catalog_subparsers = catalog.add_subparsers(
        dest="catalog_action",
        required=True,
    )
    catalog_subparsers.add_parser("audit", help="Audit dimensions and fingerprints")
    catalog_migrate = catalog_subparsers.add_parser(
        "migrate",
        help="Back up and migrate catalog metadata",
    )
    catalog_migrate.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; otherwise the command is a dry run",
    )

    paths = subparsers.add_parser("paths", help="Audit or migrate local paths")
    path_subparsers = paths.add_subparsers(dest="path_action", required=True)
    path_subparsers.add_parser("audit", help="Report non-portable and missing paths")
    migrate = path_subparsers.add_parser("migrate", help="Normalize verified paths")
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; otherwise the command is a dry run",
    )

    faiss = subparsers.add_parser("faiss", help="Build or audit exact FAISS indexes")
    faiss_subparsers = faiss.add_subparsers(dest="faiss_action", required=True)
    faiss_build = faiss_subparsers.add_parser(
        "build",
        help="Build category-partitioned exact indexes",
    )
    faiss_build.add_argument(
        "--engine",
        choices=("cosine", "metric", "all"),
        default="all",
    )
    faiss_build.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace existing indexes",
    )
    faiss_audit = faiss_subparsers.add_parser(
        "audit",
        help="Validate index manifests and files",
    )
    faiss_audit.add_argument(
        "--engine",
        choices=("cosine", "metric", "all"),
        default="all",
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare exact FAISS retrieval with the SQLite linear scan",
    )
    benchmark_subparsers = benchmark.add_subparsers(
        dest="benchmark_target",
        required=True,
    )
    benchmark_subparsers.add_parser(
        "retrieval",
        help="Benchmark vector search and verify exact parity",
    )
    api_benchmark = benchmark_subparsers.add_parser(
        "api",
        help="Benchmark complete FastAPI recommendation requests",
    )
    api_benchmark.add_argument(
        "--engine",
        choices=("cosine", "metric"),
        default=None,
        help="Recommendation engine; defaults to RECOMMENDER_ENGINE",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "catalog":
        return catalog_command(
            settings,
            action=args.catalog_action,
            apply=getattr(args, "apply", False),
        )
    if args.command == "faiss":
        return faiss_command(
            settings,
            action=args.faiss_action,
            engine=args.engine,
            replace=getattr(args, "replace", False),
        )
    if args.command == "benchmark":
        return benchmark_command(
            settings,
            target=args.benchmark_target,
            engine=getattr(args, "engine", None),
        )
    return paths_command(
        settings,
        action=args.path_action,
        apply=getattr(args, "apply", False),
    )


if __name__ == "__main__":
    sys.exit(main())
