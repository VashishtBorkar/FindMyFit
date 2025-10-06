import itertools
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from pathlib import Path
import heapq

from ..core.interfaces import RecommendationEngine
from ..core.models import (
    ClothingItem, ClothingRecommendation, OutfitRecommendation,
)
from ..services.embedding_generator import CLIPEmbeddingGenerator
from ..utils.logging import get_logger


class CosineSimilarityRecommendationEngine(RecommendationEngine):
    """Recommendations using cosine similarity of images embeddings"""
    
    def __init__(self, embeddings_dir: str = None, images_dir: str = None):
        self.logger = get_logger(__name__)
        self.embedding_generator = CLIPEmbeddingGenerator()
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.images_dir = Path(images_dir) if images_dir else None
        self.clip_embeddings = {}
    
        # self._load_embeddings()
        
    def _load_embeddings(self, match_categories: List[str]) -> None:
        """Load all precomputed embeddings into memory"""
        self.logger.info("Loading precomputed embeddings...")
        self.logger.info(f"Categories to load: {match_categories}")
        for category in match_categories:
            category_dir = self.embeddings_dir / category
            if not category_dir.exists():
                self.logger.warning(f"Embeddings directory for category '{category}' does not exist: {category_dir}")
                continue
            if category in self.clip_embeddings:
                continue  # already loaded
            self.clip_embeddings[category] = {}
            for emb_file in category_dir.glob("*.npy"):
                item_id = emb_file.stem
                self.clip_embeddings[category][item_id] = np.load(emb_file)
        
        for category, items in self.clip_embeddings.items():
            self.logger.info(f"Loaded {len(items)} embeddings for category '{category}'")

    def calculate_compatibility_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        score = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        # Scale to 0-1
        return float((score + 1) / 2)

    def get_recommendations(self, target_item: ClothingItem, max_recommendations: int, 
                            match_categories: List[str]) -> OutfitRecommendation:
        """Rank available items by similarity to the target item"""
        if not match_categories:
            raise ValueError("At least one match category must be specified")
        
        self._load_embeddings(match_categories)
        if not target_item.image_path.exists():
            raise ValueError(f"No image found at {target_item.image_path}")
        
        # Compute embedding for target image
        target_emb = self.embedding_generator.generate_embedding(str(target_item.image_path))

        # Score all items in match categories
        self.logger.info(f"Target Item ID: {target_item.id}, Category: {target_item.category}")
        self.logger.info(f"Finding matches in categories: {match_categories}")

        scored_items = []
        counter = itertools.count()
        for category in match_categories:
            for item_id, emb in self.clip_embeddings[category].items():
                # Skip same item
                if target_item.id and item_id == target_item.id:
                    continue 
                image_path = self.images_dir / category / f"{item_id}"
                
                score = self.calculate_compatibility_score(target_emb, emb)
                recommended_item = ClothingItem(
                    id=item_id,
                    image_path=image_path,
                    category=category
                )

                # Maintain max-heap of max_recommendation items
                heapq.heappush(scored_items, (score, next(counter), recommended_item))
                self.logger.info(f"({counter}) Scored Item ID : {item_id}, Category: {category}, Score: {score:.4f}")
                if len(scored_items) > max_recommendations:
                    heapq.heappop(scored_items)

        recommendations = [
            ClothingRecommendation(
                recommended_item=recommended_item,
                confidence_score=score
            )
            for score, _, recommended_item in sorted(scored_items, reverse=True)
        ]
        
        return recommendations

class MetricLearningRecommendationEngine(RecommendationEngine):
    """Recommendations using metric learning for compatibility scoring"""
    
    def __init__(self):
        pass
    
    def calculate_compatibility_score(self, item1, item2):
        return 0.0
    
    def get_recommendations(self, target_item: ClothingItem, max_recommendations: int, 
                            match_categories: List[str]) -> List[ClothingRecommendation]:
        return []


class BiLSTMRecommendationEngine(RecommendationEngine):
    """Recommendations using BiLSTM for full outfit generation"""
    
    def __init__(self):
        pass
    
    def calculate_compatibility_score(self, item1, item2):
        return 0.0
    
    def get_recommendations(self, target_item: ClothingItem, max_recommendations: int, 
                            match_categories: List[str]) -> List[ClothingRecommendation]:
        return []        