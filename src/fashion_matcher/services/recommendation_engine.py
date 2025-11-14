import itertools
import numpy as np
from typing import List, Dict, Tuple
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
from src.data_manager.embedding_manager import load_embeddings

from src.models.metric_learning.model import FashionCompatibilityModel

from src.utils.logging import get_logger


class CosineSimilarityRecommendationEngine(RecommendationEngine):
    """Recommendations using cosine similarity of images embeddings"""
    
    def __init__(self, embeddings_dir: str, images_dir: str, embeddings_pickle: str = None):
        self.logger = get_logger(__name__)
        self.embedding_generator = CLIPEmbeddingGenerator()
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.clip_embeddings, self.category_index = load_embeddings(embeddings_dir, embeddings_pickle) 
        
        self.images_dir = Path(images_dir) if images_dir else None
    
    def calculate_compatibility_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        score = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        # Scale to 0-1
        return float((score + 1) / 2)

    def get_recommendations(self, target_item: ClothingItem, max_recommendations: int, 
                            match_categories: List[str]) -> OutfitRecommendation:
        """Rank available items by similarity to the target item"""
        if not match_categories:
            raise ValueError("At least one match category must be specified")
        
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
            if category not in self.category_index:
                self.logger.warning(f"No embeddings found for category: {category}")
                continue
            for item_id in self.category_index[category]:
                # Skip same item
                if target_item.id and item_id == target_item.id:
                    continue 
                image_path = self.images_dir / category / f"{item_id}"
                
                score = self.calculate_compatibility_score(target_emb, self.clip_embeddings[item_id]["embedding"])
                recommended_item = ClothingItem(
                    id=item_id,
                    image_path=image_path,
                    category=category
                )

                scored_items.append((score, next(counter), recommended_item))
                self.logger.debug(f"({counter}) Scored Item ID : {item_id}, Category: {category}, Score: {score:.4f}")

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
    
    def __init__(self, embeddings_dir: str, images_dir: str, embeddings_pickle: str = None):
        self.logger = get_logger(__name__)
        self.clip_embeddings_generator = CLIPEmbeddingGenerator()

        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.embeddings, self.category_index = load_embeddings(embeddings_dir, embeddings_pickle) 
        
        self.images_dir = Path(images_dir) if images_dir else None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        checkpoint = torch.load('checkpoints/metric_learning/best_model.pt', map_location=device)
        self.model = FashionCompatibilityModel(embedding_dim=512)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    
    def calculate_compatibility_score(self, emb1, emb2):
        dist = np.linalg.norm(emb1 - emb2)
        score = 1 / (1 + dist)
        return float(score)
    
    def get_recommendations(self, target_item: ClothingItem, max_recommendations: int, 
                            match_categories: List[str]) -> List[ClothingRecommendation]:
        """Rank available items by similarity to the target item"""
        if not match_categories:
            raise ValueError("At least one match category must be specified")
        
        if not target_item.image_path.exists():
            raise ValueError(f"No image found at {target_item.image_path}")
        

        
        # Compute embedding for target image
        target_clip = self.clip_embeddings_generator.generate_embedding(str(target_item.image_path))
        target_emb = self.model(torch.tensor(target_clip, device=self.device).unsqueeze(0)).detach().cpu().numpy()

        # Score all items in match categories
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
                # outfit_idx, item_idx = item_id.split('_')
                image_path = self.images_dir / category / f"{item_id}"
                
                score = self.calculate_compatibility_score(target_emb, self.embeddings[item_id]["embedding"])
                recommended_item = ClothingItem(
                    id=item_id,
                    image_path=image_path,
                    category=category
                )

                scored_items.append((score, next(counter), recommended_item))
                self.logger.debug(f"({counter}) Scored Item ID : {item_id}, Category: {category}, Score: {score:.4f}")

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