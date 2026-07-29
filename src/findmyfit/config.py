"""Typed runtime configuration for FindMyFit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: Path, project_root: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _normalize_database_url(value: str, project_root: Path) -> str:
    prefix = "sqlite:///"
    if not value.startswith(prefix) or value.startswith("sqlite:////"):
        return value

    raw_path = value[len(prefix) :]
    # SQLAlchemy emits Windows absolute SQLite URLs as sqlite:///C:/...
    if len(raw_path) >= 3 and raw_path[1] == ":" and raw_path[2] == "/":
        return value

    resolved = _resolve_path(Path(raw_path), project_root)
    return f"{prefix}{resolved.as_posix()}"


class Settings(BaseSettings):
    """Configuration loaded once by an application or command entry point."""

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)
    database_url: str = "sqlite:///data/findmyfit.db"
    images_dir: Path = Path("data/images")
    metric_checkpoint_path: Path = Path("checkpoints/metric_learning/best_model.pt")
    faiss_index_dir: Path = Path("data/indexes/faiss")
    compatibility_outfits_file: Path = Field(
        default=Path("data/outfits.txt"),
        validation_alias=AliasChoices(
            "COMPATIBILITY_OUTFITS_FILE",
            "COMPATIBILE_OUTFITS_FILE",
        ),
    )
    compatibility_pairs_path: Path = Path("data/compatibility_pairs.pkl")
    optuna_storage_path: Path = Path("checkpoints/metric_learning/optuna_study.db")

    recommender_engine: Literal["cosine", "metric"] = "metric"
    retrieval_backend: Literal["faiss", "sqlite"] = "faiss"
    clip_model_name: str = "ViT-B/32"
    clip_catalog_model_name: str = "clip"
    clip_model_version: str = "vit-b32"
    metric_model_name: str = "findmyfit"
    metric_model_version: str = "v1"

    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    max_upload_bytes: int = 10 * 1024 * 1024
    max_recommendations: int = 20

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def resolve_filesystem_values(self) -> Settings:
        root = self.project_root.expanduser().resolve()
        self.project_root = root
        self.images_dir = _resolve_path(self.images_dir, root)
        self.metric_checkpoint_path = _resolve_path(self.metric_checkpoint_path, root)
        self.faiss_index_dir = _resolve_path(self.faiss_index_dir, root)
        self.compatibility_outfits_file = _resolve_path(
            self.compatibility_outfits_file, root
        )
        self.compatibility_pairs_path = _resolve_path(
            self.compatibility_pairs_path, root
        )
        self.optuna_storage_path = _resolve_path(self.optuna_storage_path, root)
        self.database_url = _normalize_database_url(self.database_url, root)
        return self

    @classmethod
    def from_env(cls, env_file: str | Path | None = None, **overrides: object) -> Settings:
        selected_env = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        return cls(_env_file=selected_env, _env_file_encoding="utf-8", **overrides)
