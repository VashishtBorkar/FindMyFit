from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


class ClothingCategory(Enum):
    """Clothing categories for classification."""
    BAG = "bag"
    BRACELET = "bracelet"
    BROOCH = "brooch"
    DRESS = "dress"
    EARRING = "earring"
    EYEWEAR = "eyewear"
    GLOVES = "gloves"
    HARIWEAR = "hairwear"
    HATS = "hats"
    JUMPSUIT = "jumpsuit"
    LEGWEAR = "legwear"
    NECKLACE = "necklace"
    NECKWEAR = "neckwear"
    OUTERWEAR = "outerwear"
    PANTS = "pants"
    RINGS = "rings"
    SHOES = "shoes"
    SKIRT = "skirt"
    TOP = "top"
    WATCHES = "watches"


class Style(Enum):
    """Style categories."""
    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    SPORTY = "sporty"
    VINTAGE = "vintage"


class Season(Enum):
    """Seasonal categories."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


@dataclass
class Color:
    """Represents a color with RGB values and optional name."""
    r: int
    g: int
    b: int
    name: Optional[str] = None
    
    def to_hex(self) -> str:
        """Convert to hex string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)


@dataclass
class ClothingItem:
    """Represents a single clothing item."""
    id: str
    image_path: Path
    category: ClothingCategory
    embedding: Optional[np.ndarray] = None
    embedding_path: Optional[Path] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure image_path is a Path object."""
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)

@dataclass
class ClothingRecommendation:
    """Represents a recommended outfit combination."""
    recommended_item: ClothingItem
    confidence_score: float
    reasoning: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")
        
@dataclass
class OutfitRecommendation:
    """Represents a recommended outfit combination."""
    target_item: ClothingItem
    recommended_items: List[ClothingItem]
    confidence_score: float
    reasoning: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")


@dataclass
class RecommendationRequest:
    """Request for outfit recommendations."""
    target_item: ClothingItem
    max_recommendations: int = 5
    style_preference: Optional[Style] = None
    season_preference: Optional[Season] = None
    match_categories: List[ClothingCategory] = field(default_factory=list)