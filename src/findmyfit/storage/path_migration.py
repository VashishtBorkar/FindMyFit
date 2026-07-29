"""Audit and explicitly normalize portable catalog and artifact paths."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from findmyfit.config import Settings
from findmyfit.db.models import Image
from findmyfit.errors import ConfigurationError
from findmyfit.storage.local_images import LocalImageStore


@dataclass
class PathAudit:
    total_rows: int = 0
    absolute_rows: int = 0
    normalized_rows: int = 0
    missing_rows: int = 0
    outside_root_rows: int = 0
    details: list[str] = field(default_factory=list)


def audit_catalog_paths(
    session_factory: sessionmaker,
    image_store: LocalImageStore,
) -> PathAudit:
    audit = PathAudit()
    with session_factory() as session:
        rows = session.query(Image.id, Image.file_path).all()

    for item_id, stored_path in rows:
        audit.total_rows += 1
        if Path(stored_path).is_absolute():
            audit.absolute_rows += 1
        try:
            key = image_store.normalize_key(stored_path)
            if key == stored_path:
                audit.normalized_rows += 1
            if not image_store.resolve(key, must_exist=False).is_file():
                audit.missing_rows += 1
                audit.details.append(f"{item_id}: missing '{key}'")
        except ConfigurationError:
            audit.outside_root_rows += 1
            audit.details.append(f"{item_id}: path is outside IMAGES_DIR")
    return audit


def sqlite_path_from_url(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ConfigurationError("Path migration currently supports SQLite only")
    return Path(database_url[len(prefix) :]).resolve()


def normalize_catalog_paths(
    session_factory: sessionmaker,
    image_store: LocalImageStore,
    database_url: str,
    *,
    apply: bool,
) -> tuple[PathAudit, Path | None, int]:
    audit = audit_catalog_paths(session_factory, image_store)
    if not apply:
        return audit, None, 0

    database_path = sqlite_path_from_url(database_url)
    if not database_path.is_file():
        raise ConfigurationError("SQLite database does not exist")
    backup_dir = database_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    backup_path = backup_dir / f"{database_path.stem}-{timestamp}{database_path.suffix}"
    shutil.copy2(database_path, backup_path)

    updated = 0
    with session_factory.begin() as session:
        for image in session.query(Image).all():
            try:
                key = image_store.normalize_key(image.file_path)
                resolved = image_store.resolve(key, must_exist=False)
            except ConfigurationError:
                continue
            if not resolved.is_file():
                continue
            if image.file_path != key:
                image.file_path = key
                updated += 1
    return audit, backup_path, updated


def legacy_artifact_pairs(settings: Settings) -> list[tuple[Path, Path]]:
    root = settings.project_root
    database_path = sqlite_path_from_url(settings.database_url)
    candidates = [
        (root / "findmyfit.db", database_path),
        (root / "best_model.pt", settings.metric_checkpoint_path),
    ]
    return [(source, target) for source, target in candidates if source != target]


def migrate_legacy_artifacts(settings: Settings, *, apply: bool) -> list[str]:
    messages: list[str] = []
    for directory_name in ("clip_embeddings", "metric_embeddings"):
        legacy_directory = settings.project_root / directory_name
        if legacy_directory.exists():
            messages.append(
                f"LEGACY: {legacy_directory} is no longer used; keep it until "
                "SQLite generation, training, and FAISS retrieval are verified"
            )
    for source, target in legacy_artifact_pairs(settings):
        if not source.exists():
            continue
        if target.exists():
            messages.append(f"CONFLICT: {source} -> {target}")
            continue
        messages.append(f"{'MOVE' if apply else 'WOULD MOVE'}: {source} -> {target}")
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    return messages
