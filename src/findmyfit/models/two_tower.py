"""Category-aware retrieval over frozen garment embeddings."""

import torch
from torch import nn
from torch.nn.functional import normalize


class TwoTowerCompatibilityModel(nn.Module):
    """Learn separate query and catalog representations for complementary search."""

    def __init__(
        self,
        embedding_dim: int,
        num_categories: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        category_dim: int = 16,
    ):
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        self.query_tower = nn.Sequential(
            nn.Linear(embedding_dim + category_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim + category_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_queries(
        self, anchors: torch.Tensor, target_categories: torch.Tensor
    ) -> torch.Tensor:
        """Encode anchors of shape [batch, embedding_dim] for the requested slots."""
        category_features = self.category_embedding(target_categories)
        inputs = torch.cat((anchors, category_features), dim=-1)
        return normalize(self.query_tower(inputs), dim=-1)

    def encode_items(
        self, items: torch.Tensor, categories: torch.Tensor
    ) -> torch.Tensor:
        """Encode catalog items with one category ID per item."""
        category_features = self.category_embedding(categories)
        inputs = torch.cat((items, category_features), dim=-1)
        return normalize(self.item_tower(inputs), dim=-1)

    def forward(
        self,
        anchors: torch.Tensor,
        target_categories: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score [batch, candidates, embedding_dim] candidate sets.

        Every candidate in a row must belong to that row's target category.
        Returns one dot-product score per candidate, with shape [batch, candidates].
        """
        queries = self.encode_queries(anchors, target_categories)
        candidate_categories = target_categories.unsqueeze(1).expand(
            -1, candidates.size(1)
        )
        items = self.encode_items(candidates, candidate_categories)
        return torch.bmm(items, queries.unsqueeze(-1)).squeeze(-1)
