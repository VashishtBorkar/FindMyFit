from pathlib import Path
from typing import List, Union

from src.fashion_matcher.core.models import (
    ClothingItem, ClothingRecommendation, OutfitRecommendation, RecommendationRequest,
)
from src.fashion_matcher.core.interfaces import RecommendationEngine 
from src.fashion_matcher.services.recommendation_engine import (
    CosineSimilarityRecommendationEngine, MetricLearningRecommendationEngine, BiLSTMRecommendationEngine
)
from src.utils.logging import get_logger


class ClothingRecommender:
    def __init__(self, recommendation_engine_type: str, embeddings_dir: str, 
                 images_dir: str = "data/images"):
        """
        Initialize Clothing Recommender
        
        Args:
            recommendation_engine_type: 'cosine' or 'metric' ('bilstm' for future)
            embeddings_dir: Directory where precomputed embeddings are stored
            images_dir: Directory where clothing item images are stored
        """
        self.logger = get_logger(__name__)
        
        # Initialize services
        self.embeddings_dir = Path(embeddings_dir)
        self.images_dir = Path(images_dir)
        self.recommendation_engine = self._initialize_recommendation_engine(recommendation_engine_type)
        
        self.logger.info("Clothing Recommender initialized successfully")
    
    def _initialize_recommendation_engine(self, engine_type: str) -> RecommendationEngine:
        """Initialize the recommendation engine."""
        if engine_type == 'cosine':
            return CosineSimilarityRecommendationEngine(embeddings_dir=self.embeddings_dir, images_dir=self.images_dir)
        
        elif engine_type == 'metric':
            # self.logger.warning(f"Metric model under construction")
            return MetricLearningRecommendationEngine(embeddings_dir=self.embeddings_dir, images_dir=self.images_dir)
        
        elif engine_type == 'bilstm':
            self.logger.warning(f"BiLSTM model under construction")
            return BiLSTMRecommendationEngine()
        
        else:
            self.logger.warning(f"Unknown engine type: {engine_type}, using cosine similarity")
            return CosineSimilarityRecommendationEngine()
    
    def get_recommendations(
        self, image_path: Union[str, Path], target_category: str,
        match_categories: List[str], max_recommendations: int = 5,
    ) -> List[ClothingRecommendation]:
        """
        Get outfit recommendations for a target item from an image path.

        Args:
            image_path: Path to the target clothing item image.
            target_category: Category of the target clothing item.
            max_recommendations: Maximum number of recommendations to return.
            exclude_categories: Categories to exclude from recommendations.

        Returns:
            List of outfit recommendations.
        """

        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            target_category = self.validate_category(target_category)
            validated_categories = [self.validate_category(cat) for cat in match_categories]

            target_item = ClothingItem(
                id=image_path.stem,  
                image_path=image_path,
                category=target_category
            )

            # Generate recommendations
            recommendations = self.recommendation_engine.get_recommendations(
                target_item=target_item,
                max_recommendations=max_recommendations,
                match_categories=validated_categories
            )

            # Extract image paths and confidence score from recommendations
            # recommended_images = [
            #     (rec.recommended_item.image_path, rec.confidence_score) for rec in recommendations
            # ]

            self.logger.info(
                f"Generated {len(recommendations)} recommendations for {image_path.name}"
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            raise
    
    @staticmethod
    def validate_category(category: str) -> str:
        allowed_categories = {
            "bag", "bracelet", "brooch", "dress", "earrings", "eyewear", "gloves",
            "hairwear", "hats", "jumpsuit", "legwear", "necklace", "neckwear", "outwear",
            "pants", "rings", "shoes", "skirt", "top", "watches"
        }
        cat = category.lower()
        if cat not in allowed_categories:
            raise ValueError(f"Unknown category '{category}'. Must be one of {allowed_categories}")
        return cat