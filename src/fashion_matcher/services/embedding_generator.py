import numpy as np
from typing import Optional
import logging
from PIL import Image
from pathlib import Path
import torch

from ..core.interfaces import EmebddingGenerator
from ..utils.logging import get_logger

class CLIPEmbeddingGenerator(EmebddingGenerator):
    """CLIP-based feature extractor for fashion items."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.device = None
        self.embedding_dim = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model."""
        try:
            import clip
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            if self.device == "cuda":
                self.model = self.model.half()
            
            self.embedding_dim = 512  # ViT-B/32
            self.logger.info(f"Initialized CLIP {self.model_name} on {self.device}")
            
        except ImportError:
            self.logger.warning("clip is required for CLIPEmbeddingGnerator")
        except Exception as e:
            self.logger.error(f"Error initializing CLIP: {str(e)}")
            raise
    
    def generate_embedding(self, image_path: Path) -> np.ndarray:
        """Create embeddings using CLIP"""
        if self.model is None:
            raise RuntimeError("CLIP model not initialized")
        
        try:
            image_path = Path(image_path)
            pil_image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding.squeeze(0).cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)
            
            self.logger.debug(f"Generated CLIP embedding: {embedding.shape}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating CLIP embedding for {image_path}: {str(e)}")
            raise

    def generate_and_save_embedding(self, image_path: Path, dir_path: Path) -> Path:
        """Generate embedding and save to embeddings_dir (as .npy)."""
        image_path = Path(image_path)
        dir_path = Path(dir_path)
        # dir_path.mkdir(parents=True, exist_ok=True)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not dir_path.exists():
            raise FileNotFoundError(f"Embedding directory does not exist: {dir_path}")

        embedding = self.generate_embedding(image_path)

        save_path = dir_path / f"{image_path.stem}.npy"
        np.save(save_path, embedding)
        self.logger.debug(f"Saved embedding to {save_path}")

        return save_path

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

class MetricEmbeddingGenerator(EmebddingGenerator):
    def __init__(self):
        pass

    def generate_embedding(self) -> np.ndarray:
        pass

    def get_embedding_dimension(self) -> int:
        pass