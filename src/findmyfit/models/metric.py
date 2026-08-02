"""Metric-learning projection model."""

import torch
from torch import nn
from torch.nn.functional import cosine_similarity, normalize


class FashionCompatibilityModel(nn.Module):
    """Project CLIP vectors into a normalized compatibility space."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return normalize(self.projector(inputs), p=2, dim=1)

    def embed_pair(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self(embedding_a), self(embedding_b)

    def compute_distance(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        distance_type: str = "euclidean",
    ) -> torch.Tensor:
        feature_a, feature_b = self.embed_pair(embedding_a, embedding_b)
        if distance_type == "euclidean":
            return torch.norm(feature_a - feature_b, dim=1)
        if distance_type == "cosine":
            return 1 - cosine_similarity(feature_a, feature_b)
        raise ValueError(f"Unknown distance type: {distance_type}")

    def predict_compatibility(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distance = self.compute_distance(embedding_a, embedding_b)
        return torch.exp(-distance), distance < threshold
