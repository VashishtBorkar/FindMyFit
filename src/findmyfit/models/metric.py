"""Metric-learning projection model."""

import torch
from torch import nn
from torch.nn.functional import cosine_similarity, normalize


class _ResidualBlock(nn.Module):
    """Two same-width linear layers with an identity skip connection."""

    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs + self.layers(inputs))


class FashionCompatibilityModel(nn.Module):
    """Project CLIP vectors into a normalized compatibility space."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        *,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.0,
        layer_norm: bool = False,
        residual_blocks: int = 0,
    ):
        super().__init__()
        widths = tuple(hidden_dims) if hidden_dims is not None else (hidden_dim,)
        if not widths or min(embedding_dim, output_dim, *widths) < 1:
            raise ValueError("Embedding and layer dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(residual_blocks, int) or residual_blocks < 0:
            raise ValueError("residual_blocks must be a nonnegative integer")
        layers = []
        input_dim = embedding_dim
        for width in widths:
            layers.append(nn.Linear(input_dim, width))
            if layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_dim = width
        layers.extend(_ResidualBlock(input_dim) for _ in range(residual_blocks))
        layers.append(nn.Linear(input_dim, output_dim))
        self.projector = nn.Sequential(*layers)

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
