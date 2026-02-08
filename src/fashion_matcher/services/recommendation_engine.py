import itertools
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import pickle
import heapq

from src.fashion_matcher.core.interfaces import RecommendationEngine
from src.fashion_matcher.core.models import (
    ClothingItem, ClothingRecommendation, OutfitRecommendation,
)
from src.fashion_matcher.services.embedding_generator import CLIPEmbeddingGenerator
from src.database.database import SessionLocal
from src.database.models import Image as ImageModel, Embedding, Model
from src.data_manager.embedding_manager import load_embeddings

from src.models.metric_learning.model import FashionCompatibilityModel

from src.utils.logging import get_logger

from typing import List, Dict, Optional, Tuple
import numpy as np

from src.database.database import SessionLocal
from src.database.models import Image as ImageModel, Embedding, Model
from src.utils.logging import get_logger


class DatabaseEmbeddingLoader:
    """Helper class to load embeddings from database."""
    
    def __init__(self, model_name: str, model_version: str):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model_version = model_version
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._category_index: Dict[str, List[str]] = {}
        self._model_id: Optional[int] = None
        
    def load_all_embeddings(self) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
        """
        Load all embeddings from database into memory for fast access.
        
        Returns:
            Tuple of (embeddings_dict, category_index)
            - embeddings_dict: {image_id: numpy_array}
            - category_index: {category: [image_id, ...]}
        """
        session = SessionLocal()
        
        try:
            # Get model ID
            model = session.query(Model).filter_by(
                name=self.model_name,
                version=self.model_version
            ).one_or_none()
            
            if not model:
                self.logger.error(f"Model {self.model_name} {self.model_version} not found in database")
                return {}, {}
            
            self._model_id = model.id
            
            # Load all embeddings with their image info
            results = session.query(
                Embedding.image_id,
                Embedding.vector,
                ImageModel.category
            ).join(
                ImageModel, Embedding.image_id == ImageModel.id
            ).filter(
                Embedding.model_id == model.id
            ).all()
            
            self.logger.info(f"Loading {len(results)} embeddings from database...")
            
            for image_id, vector_bytes, category in results:
                # Convert bytes back to numpy array
                embedding = np.frombuffer(vector_bytes, dtype=np.float32)
                self._embeddings_cache[image_id] = embedding
                
                # Build category index
                if category not in self._category_index:
                    self._category_index[category] = []
                self._category_index[category].append(image_id)
            
            self.logger.info(f"Loaded {len(self._embeddings_cache)} embeddings across {len(self._category_index)} categories")
            
            return self._embeddings_cache, self._category_index
            
        finally:
            session.close()
    
    def get_embedding(self, image_id: str) -> Optional[np.ndarray]:
        """Get a single embedding by image ID."""
        if image_id in self._embeddings_cache:
            return self._embeddings_cache[image_id]
        
        # Fallback to database query if not in cache
        session = SessionLocal()
        try:
            if self._model_id is None:
                model = session.query(Model).filter_by(
                    name=self.model_name,
                    version=self.model_version
                ).one_or_none()
                if not model:
                    return None
                self._model_id = model.id
            
            embedding = session.query(Embedding).filter_by(
                image_id=image_id,
                model_id=self._model_id
            ).one_or_none()
            
            if embedding:
                return np.frombuffer(embedding.vector, dtype=np.float32)
            return None
            
        finally:
            session.close()
    
    @property
    def embeddings(self) -> Dict[str, np.ndarray]:
        return self._embeddings_cache
    
    @property
    def category_index(self) -> Dict[str, List[str]]:
        return self._category_index

class CosineSimilarityRecommendationEngine(RecommendationEngine):
    """Recommendations using cosine similarity of image embeddings"""
    
    def __init__(self, images_dir: str):
        self.logger = get_logger(__name__)
        self.embedding_generator = CLIPEmbeddingGenerator()
        self.images_dir = Path(images_dir) if images_dir else None
        
        # Load embeddings from database
        self.embedding_loader = DatabaseEmbeddingLoader(
            model_name="clip",
            model_version="vit-b32"
        )
        self.clip_embeddings, self.category_index = self.embedding_loader.load_all_embeddings()
    
    def calculate_compatibility_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        score = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        # Scale to 0-1
        return float((score + 1) / 2)

    def get_recommendations(
        self, 
        target_item: ClothingItem, 
        max_recommendations: int, 
        match_categories: List[str]
    ) -> List[ClothingRecommendation]:
        """Rank available items by similarity to the target item"""
        if not match_categories:
            raise ValueError("At least one match category must be specified")
        
        if not target_item.image_path.exists():
            raise ValueError(f"No image found at {target_item.image_path}")
        
        # Compute embedding for target image
        target_emb = self.embedding_generator.generate_embedding(str(target_item.image_path))

        self.logger.info(f"Target Item ID: {target_item.id}, Category: {target_item.category}")
        self.logger.info(f"Finding matches in categories: {match_categories}")

        scored_items = []
        counter = itertools.count()
        
        for category in match_categories:
            if category not in self.category_index:
                self.logger.warning(f"No embeddings found for category: {category}")
                continue
                
            for item_id in self.category_index[category]:
                # Skip same item
                if target_item.id and item_id == target_item.id:
                    continue
                    
                image_path = self.images_dir / category / f"{item_id}"
                item_embedding = self.clip_embeddings.get(item_id)
                
                if item_embedding is None:
                    self.logger.warning(f"No embedding found for {item_id}")
                    continue
                
                score = self.calculate_compatibility_score(target_emb, item_embedding)
                recommended_item = ClothingItem(
                    id=item_id,
                    image_path=image_path,
                    category=category
                )

                scored_items.append((score, next(counter), recommended_item))
                self.logger.debug(f"Scored Item ID: {item_id}, Category: {category}, Score: {score:.4f}")

        top_items = heapq.nlargest(max_recommendations, scored_items)

        recommendations = [
            ClothingRecommendation(
                recommended_item=recommended_item,
                confidence_score=score
            )
            for score, _, recommended_item in top_items
        ]
        
        return recommendations


class MetricLearningRecommendationEngine(RecommendationEngine):
    """Recommendations using metric learning for compatibility scoring"""
    
    def __init__(self, images_dir: str):
        self.logger = get_logger(__name__)
        self.clip_embeddings_generator = CLIPEmbeddingGenerator()
        self.images_dir = Path(images_dir) if images_dir else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load metric embeddings from database
        self.embedding_loader = DatabaseEmbeddingLoader(
            model_name="findmyfit",
            model_version="v1"
        )
        self.embeddings, self.category_index = self.embedding_loader.load_all_embeddings()

        # Load the metric learning model for transforming new images
        checkpoint_path = "checkpoints/metric_learning/best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = FashionCompatibilityModel(
            embedding_dim=512,
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"]
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def calculate_compatibility_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        dist = np.linalg.norm(emb1 - emb2)
        score = 1 / (1 + dist)
        return float(score)
    
    def get_recommendations(
        self, 
        target_item: ClothingItem, 
        max_recommendations: int, 
        match_categories: List[str]
    ) -> List[ClothingRecommendation]:
        """Rank available items by similarity to the target item"""
        if not match_categories:
            raise ValueError("At least one match category must be specified")
        
        if not target_item.image_path.exists():
            raise ValueError(f"No image found at {target_item.image_path}")
        
        # Compute embedding for target image
        target_clip = self.clip_embeddings_generator.generate_embedding(str(target_item.image_path))
        target_emb = self.model(
            torch.tensor(target_clip, device=self.device).unsqueeze(0).float()
        ).detach().cpu().numpy().squeeze(0)

        self.logger.info(f"Target Item ID: {target_item.id}, Category: {target_item.category}")
        self.logger.info(f"Finding matches in categories: {match_categories}")
        
        scored_items = []
        counter = itertools.count()
        
        for category in match_categories:
            if category not in self.category_index:
                self.logger.warning(f"No embeddings found for category: {category}")
                continue
                
            for item_id in self.category_index[category]:
                # Skip same item
                if target_item.id and item_id == target_item.id:
                    continue
                    
                image_path = self.images_dir / category / f"{item_id}"
                item_embedding = self.embeddings.get(item_id)
                
                if item_embedding is None:
                    self.logger.warning(f"No embedding found for {item_id}")
                    continue
                
                score = self.calculate_compatibility_score(target_emb, item_embedding)
                recommended_item = ClothingItem(
                    id=item_id,
                    image_path=image_path,
                    category=category
                )

                scored_items.append((score, next(counter), recommended_item))
                self.logger.debug(f"Scored Item ID: {item_id}, Category: {category}, Score: {score:.4f}")

        top_items = heapq.nlargest(max_recommendations, scored_items)

        recommendations = [
            ClothingRecommendation(
                recommended_item=recommended_item,
                confidence_score=score
            )
            for score, _, recommended_item in top_items
        ]
        
        return recommendations


class BiLSTMRecommendationEngine(RecommendationEngine):
    """Recommendations using BiLSTM for full outfit generation"""
    
    def __init__(self):
        pass
    
    def calculate_compatibility_score(self, item1, item2):
        return 0.0
    
    def get_recommendations(self, target_item: ClothingItem, max_recommendations: int, 
                            match_categories: List[str]) -> List[ClothingRecommendation]:
        return []        