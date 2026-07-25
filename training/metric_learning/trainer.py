"""Training loop retained separately from request-time inference."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device | str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

    @staticmethod
    def calculate_metrics(
        distances: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        predictions = (distances < threshold).float()
        return {"accuracy": (predictions == labels).float().mean().item()}

    def _run_loader(
        self,
        loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        description: str,
    ) -> tuple[float, dict[str, float]]:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        distances = []
        labels_seen = []

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for embedding_a, embedding_b, labels in tqdm(
                loader, desc=description, leave=False
            ):
                embedding_a = embedding_a.to(self.device)
                embedding_b = embedding_b.to(self.device)
                labels = labels.to(self.device).float()
                if optimizer is not None:
                    optimizer.zero_grad()
                feature_a = self.model(embedding_a)
                feature_b = self.model(embedding_b)
                loss = criterion(feature_a, feature_b, labels)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                distances.append(
                    torch.sqrt(
                        torch.sum((feature_a - feature_b) ** 2, dim=1) + 1e-8
                    ).detach().cpu()
                )
                labels_seen.append(labels.detach().cpu())

        metrics = self.calculate_metrics(
            torch.cat(distances), torch.cat(labels_seen)
        )
        return total_loss / len(loader), metrics

    def train_epoch(self, loader, optimizer, criterion):
        return self._run_loader(loader, criterion, optimizer, "Training")

    def validate(self, loader, criterion):
        return self._run_loader(loader, criterion, None, "Validating")

    def test(self, loader, criterion):
        return self._run_loader(loader, criterion, None, "Testing")
