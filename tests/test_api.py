from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app
from findmyfit.config import Settings
from findmyfit.core.models import ClothingItem, ClothingRecommendation


class FakeRecommender:
    def __init__(self, **_kwargs):
        pass

    def get_recommendations(self, **_kwargs):
        image_path = TEST_IMAGE_ROOT / "shoes" / "shoe.jpg"
        return [
            ClothingRecommendation(
                ClothingItem(
                    "shoe.jpg",
                    image_path,
                    "shoes",
                    image_key="shoes/shoe.jpg",
                ),
                0.82,
            )
        ]


TEST_IMAGE_ROOT = Path()


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    return output.getvalue()


def test_recommendation_contract_does_not_expose_local_paths(tmp_path: Path):
    global TEST_IMAGE_ROOT
    TEST_IMAGE_ROOT = tmp_path / "images"
    image_path = TEST_IMAGE_ROOT / "shoes" / "shoe.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        recommender_engine="cosine",
        images_dir=TEST_IMAGE_ROOT,
        database_url=f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}",
    )
    app = create_app(settings, recommender_factory=FakeRecommender)

    with TestClient(app) as client:
        response = client.post(
            "/recommend",
            files=[
                ("image", ("target.png", _png_bytes(), "image/png")),
                ("target_category", (None, "top")),
                ("match_categories", (None, "shoes")),
                ("match_categories", (None, "pants")),
                ("max_recommendations", (None, "5")),
            ],
        )

    assert response.status_code == 200
    item = response.json()["recommendations"][0]
    assert item == {
        "item_id": "shoe.jpg",
        "category": "shoes",
        "score": 0.82,
        "image_url": "/images/shoes/shoe.jpg",
    }
    assert "image_path" not in item


def test_invalid_upload_type_is_rejected(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        recommender_engine="cosine",
        images_dir=tmp_path,
    )
    with TestClient(create_app(settings, recommender_factory=FakeRecommender)) as client:
        response = client.post(
            "/recommend",
            files=[
                ("image", ("target.txt", b"not an image", "text/plain")),
                ("target_category", (None, "top")),
                ("match_categories", (None, "shoes")),
                ("max_recommendations", (None, "5")),
            ],
        )
    assert response.status_code == 415


def test_readiness_reports_initialization_failure(tmp_path: Path):
    def failing_factory(**_kwargs):
        raise RuntimeError("secret local detail")

    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        recommender_engine="cosine",
        images_dir=tmp_path,
    )
    with TestClient(create_app(settings, recommender_factory=failing_factory)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert "secret local detail" not in response.text
