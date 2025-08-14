from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from pathlib import Path

from .models import ClothingItem, OutfitRecommendation, RecommendationRequest, Color


class ImageProcessor(ABC):
    """Abstract interface for image processing operations."""
    
    @abstractmethod
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess an image."""
        pass
    
    @abstractmethod
    def extract_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Color]:
        """Extract dominant colors from image."""
        pass
    
    @abstractmethod
    def resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image to target dimensions."""
        pass


class FeatureExtractor(ABC):
    """Abstract interface for feature extraction."""
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed image."""
        pass
    
    @abstractmethod
    def get_feature_dimension(self) -> int:
        """Get the dimension of extracted features."""
        pass


class RecommendationEngine(ABC):
    """Abstract interface for recommendation logic."""
    
    @abstractmethod
    def get_recommendations(self, request: RecommendationRequest, 
                          available_items: List[ClothingItem]) -> List[OutfitRecommendation]:
        """Generate outfit recommendations."""
        pass
    
    @abstractmethod
    def calculate_compatibility_score(self, item1: ClothingItem, 
                                    item2: ClothingItem) -> float:
        """Calculate compatibility score between two items."""
        pass


class DataStorage(ABC):
    """Abstract interface for data storage and retrieval."""
    
    @abstractmethod
    def save_item(self, item: ClothingItem) -> None:
        """Save a clothing item."""
        pass
    
    @abstractmethod
    def load_item(self, item_id: str) -> Optional[ClothingItem]:
        """Load a clothing item by ID."""
        pass
    
    @abstractmethod
    def get_all_items(self) -> List[ClothingItem]:
        """Get all stored clothing items."""
        pass
    
    @abstractmethod
    def search_items(self, **filters) -> List[ClothingItem]:
        """Search items with filters."""
        pass