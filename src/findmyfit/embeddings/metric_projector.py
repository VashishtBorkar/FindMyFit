"""Load a trained metric model and project CLIP vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from findmyfit.errors import ConfigurationError
from findmyfit.models.metric import FashionCompatibilityModel


class MetricProjector:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        embedding_dim: int = 512,
        device: str | torch.device | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not self.checkpoint_path.is_file():
            raise ConfigurationError("Metric checkpoint is missing")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model = FashionCompatibilityModel(
                embedding_dim=embedding_dim,
                hidden_dim=checkpoint["hidden_dim"],
                output_dim=checkpoint["output_dim"],
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        except Exception as exc:
            raise ConfigurationError("Metric checkpoint could not be loaded") from exc

        self.output_dim = int(checkpoint["output_dim"])

    def project(self, clip_embedding: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(
                clip_embedding, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.model(tensor).squeeze(0).cpu().numpy()

