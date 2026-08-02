"""Content hashing shared by offline catalog jobs."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
