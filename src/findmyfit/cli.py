"""Developer diagnostics and explicit path migration commands."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from findmyfit.config import Settings
from findmyfit.db.models import Model
from findmyfit.db.session import create_session_factory
from findmyfit.storage.local_images import LocalImageStore
from findmyfit.storage.path_migration import (
    audit_catalog_paths,
    migrate_legacy_artifacts,
    normalize_catalog_paths,
    sqlite_path_from_url,
)


LOGGER = logging.getLogger(__name__)


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
        database_path = sqlite_path_from_url(settings.database_url)
        if not database_path.is_file():
            raise FileNotFoundError("configured SQLite database is missing")
        session_factory = create_session_factory(settings.database_url)
        with session_factory() as session:
            session.execute(select(1))
            models = session.query(Model).all()
        model_names = ", ".join(f"{model.name}/{model.version}" for model in models)
        checks.append(_status("database", True, model_names or "connected; no models registered"))
    except Exception as exc:
        LOGGER.debug("Database doctor check failed", exc_info=True)
        checks.append(_status("database", False, type(exc).__name__))

    return 0 if all(checks) else 1


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="findmyfit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check local runtime readiness")

    paths = subparsers.add_parser("paths", help="Audit or migrate local paths")
    path_subparsers = paths.add_subparsers(dest="path_action", required=True)
    path_subparsers.add_parser("audit", help="Report non-portable and missing paths")
    migrate = path_subparsers.add_parser("migrate", help="Normalize verified paths")
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; otherwise the command is a dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.command == "doctor":
        return doctor(settings)
    return paths_command(
        settings,
        action=args.path_action,
        apply=getattr(args, "apply", False),
    )


if __name__ == "__main__":
    sys.exit(main())
