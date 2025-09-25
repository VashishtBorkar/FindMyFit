from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from .core.models import (
    ClothingItem, OutfitRecommendation, RecommendationRequest,
    ClothingCategory, Style, Season
)
from .core.interfaces import ImageProcessor, FeatureExtractor, RecommendationEngine, DataStorage
from .services.image_processor import OpenCVImageProcessor
from .services.feature_extractor import SimpleFeatureExtractor, ResNetFeatureExtractor, CLIPFeatureExtractor
from .services.recommendation_engine import RuleBasedRecommendationEngine, MLRecommendationEngine
from .services.data_manager import FileBasedDataStorage, InMemoryDataStorage
from .utils.logging import get_logger


class FashionMatcher:
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 feature_extractor_type: str = 'simple',
                 recommendation_engine_type: str = 'rule_based',
                 storage_type: str = 'file'):
        """
        Initialize Fashion Matcher
        
        Args:
            data_dir: Directory for data storage
            feature_extractor_type: 'simple', 'resnet', or 'clip'
            recommendation_engine_type: 'rule_based' or 'ml'
            storage_type: 'file' or 'memory'
        """
        self.logger = get_logger(__name__)
        
        # Initialize services
        self.image_processor = self._initialize_image_processor()
        self.feature_extractor = self._initialize_feature_extractor(feature_extractor_type)
        self.recommendation_engine = self._initialize_recommendation_engine(recommendation_engine_type)
        self.data_storage = self._initialize_storage(storage_type, data_dir)
        
        self.logger.info("Fashion Matcher initialized successfully")
    
    def _initialize_image_processor(self) -> ImageProcessor:
        return OpenCVImageProcessor()
    
    def _initialize_feature_extractor(self, extractor_type: str) -> FeatureExtractor:
        """Initialize the feature extractor."""
        if extractor_type == 'simple':
            return SimpleFeatureExtractor()
        elif extractor_type == 'resnet':
            return ResNetFeatureExtractor()
        elif extractor_type == 'clip':
            return CLIPFeatureExtractor()
        else:
            self.logger.warning(f"Unknown extractor type: {extractor_type}, using simple")
            return SimpleFeatureExtractor()
    
    def _initialize_recommendation_engine(self, engine_type: str) -> RecommendationEngine:
        """Initialize the recommendation engine."""
        if engine_type == 'rule_based':
            return RuleBasedRecommendationEngine()
        elif engine_type == 'ml':
            return MLRecommendationEngine()
        else:
            self.logger.warning(f"Unknown engine type: {engine_type}, using rule_based")
            return RuleBasedRecommendationEngine()
    
    def _initialize_storage(self, storage_type: str, data_dir: Optional[Path]) -> DataStorage:
        """Initialize the data storage."""
        if storage_type == 'file':
            return FileBasedDataStorage(data_dir or Path("data"))
        elif storage_type == 'memory':
            return InMemoryDataStorage()
        else:
            self.logger.warning(f"Unknown storage type: {storage_type}, using file")
            return FileBasedDataStorage(data_dir or Path("data"))
    
    def add_clothing_item(self, 
                         image_path: Path,
                         category: ClothingCategory,
                         style: Optional[Style] = None,
                         season: Optional[Season] = None,
                         item_id: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> ClothingItem:
        """
        Add a new clothing item to the system.
        
        Args:
            image_path: Path to the clothing image
            category: Clothing category
            style: Style category (optional)
            season: Season category (optional)
            item_id: Custom ID (optional, will generate if not provided)
            metadata: Additional metadata (optional)
        
        Returns:
            The created ClothingItem
        """
        try:
            # Generate ID if not provided
            if item_id is None:
                item_id = str(uuid.uuid4())
            
            image = self.image_processor.load_image(image_path)
            resized_image = self.image_processor.resize_image(image, (224, 224))
            embedding = self.feature_extractor.extract_features(resized_image)
            
            # Create clothing item
            item = ClothingItem(
                id=item_id,
                image_path=image_path,
                category=category,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            self.data_storage.save_item(item)

            self.logger.info(f"Added clothing item: {item_id} ({category.value})")
            return item
            
        except Exception as e:
            self.logger.error(f"Error adding clothing item: {str(e)}")
            raise
    
    def get_recommendations(self,
                          target_item_id: str,
                          max_recommendations: int = 5,
                          style_preference: Optional[Style] = None,
                          season_preference: Optional[Season] = None,
                          exclude_categories: Optional[List[ClothingCategory]] = None) -> List[OutfitRecommendation]:
        """
        Get outfit recommendations for a target item.
        
        Args:
            target_item_id: ID of the target clothing item
            max_recommendations: Maximum number of recommendations
            style_preference: Preferred style (optional)
            season_preference: Preferred season (optional)
            exclude_categories: Categories to exclude (optional)
        
        Returns:
            List of outfit recommendations
        """
        try:
            target_item = self.data_storage.load_item(target_item_id)
            if target_item is None:
                raise ValueError(f"Item not found: {target_item_id}")
            
            available_items = self.data_storage.get_all_items()
            
            # Create recommendation request
            request = RecommendationRequest(
                target_item=target_item,
                max_recommendations=max_recommendations,
                style_preference=style_preference,
                season_preference=season_preference,
                exclude_categories=exclude_categories or []
            )

            recommendations = self.recommendation_engine.get_recommendations(request, available_items)
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for {target_item_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            raise
    
    def get_item(self, item_id: str) -> Optional[ClothingItem]:
        """Get a clothing item by ID."""
        return self.data_storage.load_item(item_id)
    
    def get_all_items(self) -> List[ClothingItem]:
        """Get all clothing items."""
        return self.data_storage.get_all_items()
    
    def search_items(self, **filters) -> List[ClothingItem]:
        """Search clothing items with filters."""
        return self.data_storage.search_items(**filters)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            storage_stats = self.data_storage.get_statistics()
            
            system_stats = {
                'feature_extractor': type(self.feature_extractor).__name__,
                'recommendation_engine': type(self.recommendation_engine).__name__,
                'image_processor': type(self.image_processor).__name__,
                'feature_dimension': self.feature_extractor.get_feature_dimension(),
                **storage_stats
            }
            
            return system_stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {'error': str(e)}
    
    def calculate_item_similarity(self, item_id1: str, item_id2: str) -> float:
        """Calculate similarity score between two items."""
        try:
            item1 = self.data_storage.load_item(item_id1)
            item2 = self.data_storage.load_item(item_id2)
            
            if item1 is None or item2 is None:
                raise ValueError("One or both items not found")
            
            score = self.recommendation_engine.calculate_compatibility_score(item1, item2)
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    def batch_process_images(self, image_directory: Path, 
                           default_category: ClothingCategory,
                           default_style: Optional[Style] = None,
                           default_season: Optional[Season] = None) -> List[ClothingItem]:
        """
        Process multiple images from a directory.
        
        Args:
            image_directory: Directory containing clothing images
            default_category: Default category for all items
            default_style: Default style for all items
            default_season: Default season for all items
        
        Returns:
            List of processed ClothingItems
        """
        processed_items = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        try:
            image_paths = [
                p for p in image_directory.iterdir() 
                if p.suffix.lower() in image_extensions
            ]
            
            self.logger.info(f"Processing {len(image_paths)} images from {image_directory}")
            
            for image_path in image_paths:
                try:
                    item = self.add_clothing_item(
                        image_path=image_path,
                        category=default_category,
                        style=default_style,
                        season=default_season,
                        metadata={'source': 'batch_process'}
                    )
                    processed_items.append(item)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {image_path}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully processed {len(processed_items)} items")
            return processed_items
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise