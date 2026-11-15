"""Training objectives for garment compatibility."""

import torch
from torch import nn
from torch.nn.functional import cross_entropy


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


class CandidateRankingLoss(nn.Module):
    """Rank one observed partner within each row of sampled candidates."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be positive and finite")
        self.temperature = temperature

    def forward(
        self, scores: torch.Tensor, positive_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compare [batch, candidates] scores against [batch] partner indices."""
        return cross_entropy(scores / self.temperature, positive_indices)


class CandidateContrastiveLoss(nn.Module):
    """Apply margin-based contrastive loss to normalized-vector similarities.

    Positive pairs and sampled alternative pairs receive equal total weight,
    independent of the number of candidates in a row.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        if not 0 < margin < float("inf"):
            raise ValueError("margin must be positive and finite")
        self.margin = margin

    def forward(self, scores: torch.Tensor, positive_indices: torch.Tensor) -> torch.Tensor:
        if scores.size(1) < 2:
            raise ValueError("Contrastive training requires at least two candidates")
        # For unit vectors, squared Euclidean distance equals 2 - 2 * dot product.
        distances = ((2 - 2 * scores).clamp(min=0) + 1e-8).sqrt()
        positives = distances.gather(1, positive_indices.unsqueeze(1)).squeeze(1)
        negative_mask = torch.ones_like(scores, dtype=torch.bool)
        negative_mask.scatter_(1, positive_indices.unsqueeze(1), False)
        negative_loss = (self.margin - distances[negative_mask]).clamp(min=0).square().mean()
        return 0.5 * (positives.square().mean() + negative_loss)
