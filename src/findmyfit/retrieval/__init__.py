"""Catalog vector-search implementations."""

from findmyfit.retrieval.faiss_index import FaissCatalogSearch
from findmyfit.retrieval.sqlite_search import SqliteLinearSearch

__all__ = ["FaissCatalogSearch", "SqliteLinearSearch"]
