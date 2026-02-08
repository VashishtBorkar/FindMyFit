from pathlib import Path
from typing import List, Union

from src.fashion_matcher.core.models import ClothingItem, ClothingRecommendation
from src.fashion_matcher.core.interfaces import RecommendationEngine 
from src.fashion_matcher.services.recommendation_engine import (
    CosineSimilarityRecommendationEngine, 
    MetricLearningRecommendationEngine, 
    BiLSTMRecommendationEngine
)
from src.utils.logging import get_logger


class ClothingRecommender:
    
    ALLOWED_CATEGORIES = {
        "bag", "bracelet", "brooch", "dress", "earrings", "eyewear", "gloves",
        "hairwear", "hats", "jumpsuit", "legwear", "necklace", "neckwear", "outwear",
        "pants", "rings", "shoes", "skirt", "top", "watches"
    }
    
    def __init__(
        self, 
        recommendation_engine_type: str, 
        images_dir: str = "data/images"
    ):
        """
        Initialize Clothing Recommender.
        
        Args:
            recommendation_engine_type: 'cosine', 'metric', or 'bilstm'
            images_dir: Directory where clothing item images are stored
        """
        self.logger = get_logger(__name__)
        self.images_dir = Path(images_dir)
        self.recommendation_engine = self._initialize_recommendation_engine(recommendation_engine_type)
        
        self.logger.info("Clothing Recommender initialized successfully")
    
    def _initialize_recommendation_engine(self, engine_type: str) -> RecommendationEngine:
        """Initialize the recommendation engine."""
        engines = {
            'cosine': CosineSimilarityRecommendationEngine,
            'metric': MetricLearningRecommendationEngine,
            'bilstm': BiLSTMRecommendationEngine,
        }
        
        if engine_type not in engines:
            self.logger.warning(f"Unknown engine type: {engine_type}, using cosine similarity")
            engine_type = 'cosine'
        
        if engine_type == 'bilstm':
            self.logger.warning("BiLSTM model under construction")
        
        return engines[engine_type](images_dir=self.images_dir)
    
    def get_recommendations(
        self, 
        image_path: Union[str, Path], 
        target_category: str,
        match_categories: List[str], 
        max_recommendations: int = 5,
    ) -> List[ClothingRecommendation]:
        """
        Get outfit recommendations for a target item.

        Args:
            image_path: Path to the target clothing item image
            target_category: Category of the target clothing item
            match_categories: Categories to find recommendations from
            max_recommendations: Maximum number of recommendations to return

        Returns:
            List of clothing recommendations
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

            recommendations = self.recommendation_engine.get_recommendations(
                target_item=target_item,
                max_recommendations=max_recommendations,
                match_categories=validated_categories
            )

            self.logger.info(
                f"Generated {len(recommendations)} recommendations for {image_path.name}"
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            raise
    
    @classmethod
    def validate_category(cls, category: str) -> str:
        """Validate and normalize a category string."""
        cat = category.lower()
        if cat not in cls.ALLOWED_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Must be one of {cls.ALLOWED_CATEGORIES}"
            )
        return cat
    
    @classmethod
    def get_allowed_categories(cls) -> List[str]:
        """Return list of allowed categories."""
        return sorted(list(cls.ALLOWED_CATEGORIES))