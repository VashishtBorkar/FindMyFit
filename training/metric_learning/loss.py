"""Contrastive loss used by the metric-learning experiment."""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        distance = torch.sqrt(
            torch.sum((embedding_a - embedding_b) ** 2, dim=1) + 1e-8
        )
        positive_loss = labels * torch.pow(distance, 2)
        negative_loss = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )
        return torch.mean(positive_loss + negative_loss)
