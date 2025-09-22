import pytest
from fashion_matcher import FashionMatcher
from fashion_matcher.core.models import ClothingCategory

def test_fashion_matcher_initialization():
    """Check that the FashionMatcher can be initialized."""
    fm = FashionMatcher()
    assert fm is not None

