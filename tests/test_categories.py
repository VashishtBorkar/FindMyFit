import pytest

from findmyfit.core.categories import get_allowed_categories, normalize_category
from findmyfit.errors import InvalidCategoryError


def test_category_normalization_preserves_database_names():
    assert normalize_category("  OUTWEAR ") == "outwear"
    assert "earrings" in get_allowed_categories()
    assert "hairwear" in get_allowed_categories()


def test_unknown_category_is_rejected():
    with pytest.raises(InvalidCategoryError):
        normalize_category("outerwear")
