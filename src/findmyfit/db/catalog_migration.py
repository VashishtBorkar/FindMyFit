"""Audit and migrate SQLite catalog metadata without loading vector values."""

from __future__ import annotations

import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from findmyfit.config import Settings
from findmyfit.embeddings.fingerprints import (
    clip_artifact_fingerprint,
    metric_artifact_fingerprint,
)
from findmyfit.errors import CatalogError
from findmyfit.storage.path_migration import sqlite_path_from_url

REQUIRED_INDEXES = {
    "ix_images_category": "CREATE INDEX IF NOT EXISTS ix_images_category ON images (category)",
    "ix_embeddings_model_id": (
        "CREATE INDEX IF NOT EXISTS ix_embeddings_model_id ON embeddings (model_id)"
    ),
}


@dataclass(frozen=True)
class ModelAudit:
    model_id: int
    name: str
    version: str
    recorded_dimension: int
    expected_dimension: int | None
    declared_dimensions: tuple[int, ...]
    actual_dimensions: tuple[int, ...]
    row_count: int
    missing_items: int
    fingerprint_matches: bool

    @property
    def ready(self) -> bool:
        return (
            self.expected_dimension is not None
            and self.recorded_dimension == self.expected_dimension
            and self.declared_dimensions == (self.expected_dimension,)
            and self.actual_dimensions == (self.expected_dimension,)
            and self.fingerprint_matches
        )


@dataclass
class CatalogAudit:
    database_path: Path
    fingerprint_column: bool
    missing_indexes: list[str] = field(default_factory=list)
    models: list[ModelAudit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return (
            self.fingerprint_column
            and not self.missing_indexes
            and not self.errors
            and bool(self.models)
            and all(model.ready for model in self.models)
        )


def _checkpoint_output_dimension(path: Path) -> int:
    try:
        import torch
    except ImportError as exc:
        raise CatalogError("Install the ml extra to inspect metric metadata") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return int(checkpoint["output_dim"])


def expected_model_metadata(
    settings: Settings,
    name: str,
    version: str,
) -> tuple[int | None, str | None]:
    if (
        name == settings.clip_catalog_model_name
        and version == settings.clip_model_version
    ):
        return (
            512,
            clip_artifact_fingerprint(
                settings.clip_model_name,
                settings.clip_model_version,
            ),
        )
    if name == settings.metric_model_name and version == settings.metric_model_version:
        if not settings.metric_checkpoint_path.is_file():
            return None, None
        return (
            _checkpoint_output_dimension(settings.metric_checkpoint_path),
            metric_artifact_fingerprint(
                settings.metric_checkpoint_path,
                model_version=settings.metric_model_version,
                clip_model_version=settings.clip_model_version,
            ),
        )
    return None, None


def audit_catalog(settings: Settings) -> CatalogAudit:
    database_path = sqlite_path_from_url(settings.database_url)
    audit = CatalogAudit(database_path=database_path, fingerprint_column=False)
    if not database_path.is_file():
        audit.errors.append("configured SQLite database is missing")
        return audit

    with sqlite3.connect(database_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            audit.errors.append(f"integrity check failed: {integrity}")
            return audit

        model_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(models)")
        }
        audit.fingerprint_column = "artifact_fingerprint" in model_columns
        existing_indexes = {
            row[1]
            for table in ("images", "embeddings")
            for row in connection.execute(f"PRAGMA index_list({table})")
        }
        audit.missing_indexes = sorted(
            set(REQUIRED_INDEXES).difference(existing_indexes)
        )

        fingerprint_sql = (
            "artifact_fingerprint"
            if audit.fingerprint_column
            else "NULL AS artifact_fingerprint"
        )
        models = connection.execute(
            "SELECT id, name, version, embedding_dim, "
            f"{fingerprint_sql} FROM models ORDER BY id"
        ).fetchall()

        layouts: dict[int, Counter[int]] = defaultdict(Counter)
        declared_layouts: dict[int, Counter[int]] = defaultdict(Counter)
        for model_id, declared_dim, dtype, byte_length in connection.execute(
            "SELECT model_id, dim, dtype, length(vector) FROM embeddings"
        ):
            try:
                item_size = np.dtype(dtype).itemsize
            except (TypeError, ValueError):
                audit.errors.append(
                    f"model id {model_id} contains unsupported dtype '{dtype}'"
                )
                continue
            if byte_length % item_size:
                audit.errors.append(
                    f"model id {model_id} contains an invalid vector byte length"
                )
                continue
            declared_layouts[int(model_id)][int(declared_dim)] += 1
            layouts[int(model_id)][int(byte_length // item_size)] += 1

        image_count = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        for model_id, name, version, recorded_dim, fingerprint in models:
            try:
                expected_dim, expected_fingerprint = expected_model_metadata(
                    settings, name, version
                )
            except Exception as exc:  # noqa: BLE001 - audit must report artifact failures
                audit.errors.append(f"{name}/{version}: {type(exc).__name__}")
                expected_dim, expected_fingerprint = None, None
            row_count = sum(layouts[int(model_id)].values())
            audit.models.append(
                ModelAudit(
                    model_id=int(model_id),
                    name=name,
                    version=version,
                    recorded_dimension=int(recorded_dim),
                    expected_dimension=expected_dim,
                    declared_dimensions=tuple(
                        sorted(declared_layouts[int(model_id)])
                    ),
                    actual_dimensions=tuple(sorted(layouts[int(model_id)])),
                    row_count=row_count,
                    missing_items=max(0, image_count - row_count),
                    fingerprint_matches=(
                        expected_fingerprint is not None
                        and fingerprint == expected_fingerprint
                    ),
                )
            )
    return audit


def migrate_catalog(
    settings: Settings,
    *,
    apply: bool,
) -> tuple[CatalogAudit, Path | None]:
    before = audit_catalog(settings)
    if not apply:
        return before, None
    if before.errors:
        raise CatalogError("; ".join(before.errors))

    database_path = before.database_path
    backup_dir = database_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    backup_path = backup_dir / (
        f"{database_path.stem}-before-catalog-{timestamp}{database_path.suffix}"
    )
    shutil.copy2(database_path, backup_path)

    try:
        with sqlite3.connect(database_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(models)")
            }
            if "artifact_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE models ADD COLUMN artifact_fingerprint VARCHAR"
                )
            for statement in REQUIRED_INDEXES.values():
                connection.execute(statement)

            models = connection.execute(
                "SELECT id, name, version, artifact_fingerprint "
                "FROM models ORDER BY id"
            ).fetchall()
            for model_id, name, version, current_fingerprint in models:
                expected_dim, expected_fingerprint = expected_model_metadata(
                    settings, name, version
                )
                if expected_dim is None or expected_fingerprint is None:
                    raise CatalogError(
                        f"Cannot derive metadata for model '{name}/{version}'"
                    )
                if (
                    current_fingerprint
                    and current_fingerprint != expected_fingerprint
                ):
                    raise CatalogError(
                        f"Model '{name}/{version}' has an incompatible "
                        "artifact fingerprint"
                    )
                layouts = {
                    int(byte_length) // np.dtype(dtype).itemsize
                    for dtype, byte_length in connection.execute(
                        "SELECT dtype, length(vector) FROM embeddings "
                        "WHERE model_id = ?",
                        (model_id,),
                    )
                }
                if layouts != {expected_dim}:
                    raise CatalogError(
                        f"Stored vectors for '{name}/{version}' have dimensions "
                        f"{sorted(layouts)}; expected {expected_dim}"
                    )
                connection.execute(
                    "UPDATE embeddings SET dim = ? WHERE model_id = ?",
                    (expected_dim, model_id),
                )
                connection.execute(
                    "UPDATE models SET embedding_dim = ?, artifact_fingerprint = ? "
                    "WHERE id = ?",
                    (expected_dim, expected_fingerprint, model_id),
                )
    except Exception:
        shutil.copy2(backup_path, database_path)
        raise
    return audit_catalog(settings), backup_path
