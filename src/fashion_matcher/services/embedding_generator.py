import numpy as np
from typing import Optional
import logging
from PIL import Image
from pathlib import Path
import torch

from src.fashion_matcher.core.interfaces import EmebddingGenerator
from src.utils.logging import get_logger
from src.database.database import SessionLocal
from src.database.models import Image as ImageModel, Embedding, Model

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

    def _get_or_create_model(self, session, name: str, version: str, embedding_dim: int) -> int:
        """Get existing model ID or create new model record."""
        model = session.query(Model).filter_by(
            name=name, 
            version=version
        ).one_or_none()
        
        if model:
            return model.id
        
        model = Model(
            name=name,
            version=version,
            embedding_dim=embedding_dim
        )
        session.add(model)
        session.commit()
        return model.id

    def generate_and_save_embedding(
        self, 
        image_path: Path, 
        image_id: str = None,
        model_name: str = "clip",
        model_version: str = "vit-b32",
        embedding_dim: int = 512
    ) -> np.ndarray:
        """
        Generate embedding and save to database.
        
        Args:
            image_path: Path to the image file
            image_id: ID to use for the image (defaults to filename stem)
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            embedding_dim: Dimension of the embedding vector
            
        Returns:
            The generated embedding as numpy array
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use filename stem as ID if not provided
        if image_id is None:
            image_id = image_path.stem
        
        session = SessionLocal()
        
        try:
            # Get or create the model record
            model_id = self._get_or_create_model(
                session, model_name, model_version, embedding_dim
            )
            
            # Check if embedding already exists
            existing_embedding = session.query(Embedding).filter_by(
                image_id=image_id,
                model_id=model_id
            ).one_or_none()
            
            if existing_embedding:
                self.logger.debug(f"Embedding already exists for {image_id}, skipping")
                # Return existing embedding as numpy array
                return np.frombuffer(
                    existing_embedding.vector, 
                    dtype=np.float32
                )
            
            # Generate new embedding
            embedding = self.generate_embedding(image_path)
            
            # Convert to bytes for storage
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            # Save to database
            db_embedding = Embedding(
                image_id=image_id,
                model_id=model_id,
                vector=embedding_bytes,
                dim=embedding_dim,
                dtype="float32"
            )
            session.add(db_embedding)
            session.commit()
            
            self.logger.debug(f"Saved embedding to database for {image_id}")
            return embedding
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving embedding for {image_id}: {str(e)}")
            raise
        finally:
            session.close()

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

class MetricEmbeddingGenerator(EmebddingGenerator):
    def __init__(self):
        pass

    def generate_embedding(self) -> np.ndarray:
        pass

    def get_embedding_dimension(self) -> int:
        pass