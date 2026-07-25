from pathlib import Path

from findmyfit.config import Settings


def test_settings_resolve_relative_paths_from_project_root(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        database_url="sqlite:///data/catalog.db",
        images_dir=Path("catalog/images"),
        metric_checkpoint_path=Path("models/model.pt"),
    )

    assert settings.images_dir == (tmp_path / "catalog/images").resolve()
    assert settings.metric_checkpoint_path == (tmp_path / "models/model.pt").resolve()
    assert settings.database_url == f"sqlite:///{(tmp_path / 'data/catalog.db').as_posix()}"


def test_settings_parse_comma_separated_cors_origins(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        cors_origins="http://localhost:5173,https://example.test",
    )
    assert settings.cors_origins == [
        "http://localhost:5173",
        "https://example.test",
    ]
