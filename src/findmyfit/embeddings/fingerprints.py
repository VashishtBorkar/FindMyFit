"""Stable fingerprints for versioned embedding artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

CLIP_PREPROCESSING_VERSION = "openai-clip-rgb-normalized-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_artifact_fingerprint(model_name: str, model_version: str) -> str:
    descriptor = (
        f"clip:{model_name}:{model_version}:{CLIP_PREPROCESSING_VERSION}"
    )
    return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()


def metric_artifact_fingerprint(
    checkpoint_path: str | Path,
    *,
    model_version: str,
    clip_model_version: str,
) -> str:
    descriptor = (
        f"metric:{model_version}:{clip_model_version}:"
        f"{sha256_file(checkpoint_path)}"
    )
    return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()
