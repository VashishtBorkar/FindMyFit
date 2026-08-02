import pytest

from findmyfit.clothing_recommender import ClothingRecommender
from findmyfit.config import Settings
from findmyfit.errors import ConfigurationError


def test_facade_rejects_removed_or_unknown_engines():
    with pytest.raises(ConfigurationError):
        ClothingRecommender("bilstm")
    with pytest.raises(ConfigurationError):
        ClothingRecommender("unknown")


@pytest.mark.parametrize("backend", ["sqlite", "faiss"])
def test_facade_selects_configured_retrieval_backend(
    tmp_path,
    monkeypatch,
    backend,
):
    selected = []

    class FakeRepository:
        def __init__(self, *_args):
            pass

    class FakeSearch:
        def __init__(self, *_args, **_kwargs):
            selected.append(backend)

    class UnexpectedSearch:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("Wrong retrieval backend selected")

    class FakeEmbedder:
        def __init__(self, *_args):
            pass

    monkeypatch.setattr(
        "findmyfit.clothing_recommender.SqliteEmbeddingRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        "findmyfit.clothing_recommender.ClipEmbedder",
        FakeEmbedder,
    )
    monkeypatch.setattr(
        "findmyfit.clothing_recommender.create_session_factory",
        lambda _url: object(),
    )
    monkeypatch.setattr(
        "findmyfit.clothing_recommender.SqliteLinearSearch",
        FakeSearch if backend == "sqlite" else UnexpectedSearch,
    )
    monkeypatch.setattr(
        "findmyfit.clothing_recommender.FaissCatalogSearch",
        FakeSearch if backend == "faiss" else UnexpectedSearch,
    )
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        images_dir=tmp_path,
        retrieval_backend=backend,
        recommender_engine="cosine",
    )
    ClothingRecommender("cosine", settings=settings)
    assert selected == [backend]
