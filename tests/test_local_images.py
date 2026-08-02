from pathlib import Path

import pytest

from findmyfit.errors import ConfigurationError
from findmyfit.storage.local_images import LocalImageStore


def test_local_image_store_resolves_portable_keys(tmp_path: Path):
    image = tmp_path / "shoes" / "shoe.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    store = LocalImageStore(tmp_path)

    assert store.resolve("shoes/shoe.jpg") == image
    assert store.normalize_key(image) == "shoes/shoe.jpg"
    assert store.url_for("shoes/shoe.jpg") == "/images/shoes/shoe.jpg"


@pytest.mark.parametrize("key", ["../secret.jpg", "/tmp/secret.jpg", "C:/secret.jpg"])
def test_local_image_store_rejects_unsafe_keys(tmp_path: Path, key: str):
    store = LocalImageStore(tmp_path)
    with pytest.raises(ConfigurationError):
        store.resolve(key, must_exist=False)


def test_local_image_store_reports_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LocalImageStore(tmp_path).resolve("shoes/missing.jpg")
