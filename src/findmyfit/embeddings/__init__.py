"""Runtime embedding components, imported lazily to avoid loading ML dependencies."""

from typing import Any


__all__ = ["ClipEmbedder", "MetricProjector"]


def __getattr__(name: str) -> Any:
    if name == "ClipEmbedder":
        from findmyfit.embeddings.clip import ClipEmbedder

        return ClipEmbedder
    if name == "MetricProjector":
        from findmyfit.embeddings.metric_projector import MetricProjector

        return MetricProjector
    raise AttributeError(name)

