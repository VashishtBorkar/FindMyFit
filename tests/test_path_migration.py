from pathlib import Path

from findmyfit.db.models import Base, Image
from findmyfit.db.session import create_engine_for_url, create_session_factory
from findmyfit.storage.local_images import LocalImageStore
from findmyfit.storage.path_migration import (
    audit_catalog_paths,
    normalize_catalog_paths,
)


def test_catalog_path_migration_is_dry_run_then_idempotent(tmp_path: Path):
    images_root = tmp_path / "images"
    image_path = images_root / "top" / "item.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    database_url = f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}"
    Base.metadata.create_all(create_engine_for_url(database_url))
    session_factory = create_session_factory(database_url)
    with session_factory.begin() as session:
        session.add(
            Image(
                id="item.jpg",
                file_path=str(image_path),
                category="top",
                hash="hash",
            )
        )

    audit = audit_catalog_paths(session_factory, LocalImageStore(images_root))
    assert audit.absolute_rows == 1

    _, backup, updated = normalize_catalog_paths(
        session_factory,
        LocalImageStore(images_root),
        database_url,
        apply=False,
    )
    assert backup is None
    assert updated == 0

    _, backup, updated = normalize_catalog_paths(
        session_factory,
        LocalImageStore(images_root),
        database_url,
        apply=True,
    )
    assert backup is not None and backup.is_file()
    assert updated == 1
    with session_factory() as session:
        assert session.get(Image, "item.jpg").file_path == "top/item.jpg"

    _, _, updated_again = normalize_catalog_paths(
        session_factory,
        LocalImageStore(images_root),
        database_url,
        apply=True,
    )
    assert updated_again == 0
