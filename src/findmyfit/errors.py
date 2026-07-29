"""Application-level exceptions that are independent of HTTP or UI frameworks."""


class FindMyFitError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(FindMyFitError):
    """Raised when required runtime configuration or an artifact is unavailable."""


class InvalidCategoryError(FindMyFitError, ValueError):
    """Raised when a request contains an unsupported clothing category."""


class InvalidRecommendationRequest(FindMyFitError, ValueError):
    """Raised when recommendation request values are invalid."""


class CatalogError(FindMyFitError):
    """Raised when catalog metadata or embeddings cannot be loaded."""


class VectorIndexError(FindMyFitError):
    """Raised when a vector-search index is missing, stale, or invalid."""
