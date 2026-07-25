import pytest

from findmyfit.clothing_recommender import ClothingRecommender
from findmyfit.errors import ConfigurationError


def test_facade_rejects_removed_or_unknown_engines():
    with pytest.raises(ConfigurationError):
        ClothingRecommender("bilstm")
    with pytest.raises(ConfigurationError):
        ClothingRecommender("unknown")
