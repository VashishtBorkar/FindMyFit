"""CLIP image inference without persistence concerns."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from findmyfit.errors import ConfigurationError


LOGGER = logging.getLogger(__name__)


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None):
        try:
            import clip
            import torch
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAI CLIP is not installed. Install the 'ml' project extra."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        except Exception as exc:
            raise ConfigurationError(f"Unable to initialize CLIP model '{model_name}'") from exc
        self.embedding_dim = 512
        LOGGER.info("Initialized CLIP %s on %s", model_name, self.device)

    def embed(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path.name}")

        with Image.open(path) as source:
            image = source.convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            vector = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("CLIP produced a zero-length embedding")
        return vector / norm

    # Compatibility with the previous method name used by scripts.
    generate_embedding = embed
