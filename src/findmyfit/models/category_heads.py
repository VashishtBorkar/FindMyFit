"""Shared visual features with a separate compatibility geometry per target category."""

import torch
from torch import nn
from torch.nn.functional import normalize


class CategoryHeadCompatibilityModel(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256, output_dim=128,
                 category_ids=(14, 16)):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU())
        self.heads = nn.ModuleDict({str(c): nn.Linear(hidden_dim, output_dim) for c in category_ids})

    def encode(self, features, categories):
        hidden = self.backbone(features)
        output = hidden.new_zeros((len(hidden), next(iter(self.heads.values())).out_features))
        for category in categories.unique():
            mask = categories == category
            output[mask] = normalize(self.heads[str(category.item())](hidden[mask]), dim=-1)
        return output

    def forward(self, anchors, categories, candidates):
        queries = self.encode(anchors, categories)
        items = self.encode(candidates.flatten(0, 1), categories[:, None].expand(
            -1, candidates.size(1)).reshape(-1)).reshape(candidates.size(0), candidates.size(1), -1)
        return torch.bmm(items, queries.unsqueeze(-1)).squeeze(-1)
