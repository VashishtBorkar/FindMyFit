"""Safe mapping between portable catalog keys and local image files."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import quote

from findmyfit.errors import ConfigurationError


class LocalImageStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def normalize_key(self, value: str | Path) -> str:
        raw_path = Path(value)
        if raw_path.is_absolute():
            try:
                raw_path = raw_path.resolve().relative_to(self.root)
            except ValueError as exc:
                raise ConfigurationError("Catalog image path is outside IMAGES_DIR") from exc

        posix = PurePosixPath(raw_path.as_posix())
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise ConfigurationError("Catalog image key must stay beneath IMAGES_DIR")
        return posix.as_posix()

    def resolve(self, key: str | Path, *, must_exist: bool = True) -> Path:
        normalized = self.normalize_key(key)
        resolved = (self.root / Path(normalized)).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ConfigurationError("Catalog image key escapes IMAGES_DIR") from exc
        if must_exist and not resolved.is_file():
            raise FileNotFoundError(f"Catalog image does not exist: {normalized}")
        return resolved

    def url_for(self, key: str | Path) -> str:
        normalized = self.normalize_key(key)
        encoded = "/".join(quote(part) for part in PurePosixPath(normalized).parts)
        return f"/images/{encoded}"
