"""Canonical category values used by the database and recommendation API."""

from findmyfit.errors import InvalidCategoryError


ALLOWED_CATEGORIES = frozenset(
    {
        "bag",
        "bracelet",
        "brooch",
        "dress",
        "earrings",
        "eyewear",
        "gloves",
        "hairwear",
        "hats",
        "jumpsuit",
        "legwear",
        "necklace",
        "neckwear",
        "outwear",
        "pants",
        "rings",
        "shoes",
        "skirt",
        "top",
        "watches",
    }
)


def normalize_category(category: str) -> str:
    normalized = category.strip().lower()
    if normalized not in ALLOWED_CATEGORIES:
        choices = ", ".join(sorted(ALLOWED_CATEGORIES))
        raise InvalidCategoryError(
            f"Unknown category '{category}'. Expected one of: {choices}"
        )
    return normalized


def get_allowed_categories() -> list[str]:
    return sorted(ALLOWED_CATEGORIES)
