import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import colorsys

from ..core.interfaces import RecommendationEngine
from ..core.models import (
    ClothingItem, OutfitRecommendation, RecommendationRequest, 
    ClothingCategory, Style, Color
)
from ..utils.logging import get_logger


class RuleBasedRecommendationEngine(RecommendationEngine):
    """Rule-based recommendation engine using fashion compatibility rules."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Define compatibility rules
        self.category_compatibility = self._initialize_category_rules()
        self.style_compatibility = self._initialize_style_rules()
        self.color_harmony_rules = self._initialize_color_rules()
    
    def _initialize_category_rules(self) -> Dict[ClothingCategory, List[ClothingCategory]]:
        """Initialize category compatibility rules."""
        return {
            ClothingCategory.TOPS: [
                ClothingCategory.BOTTOMS, ClothingCategory.SHOES, 
                ClothingCategory.OUTERWEAR, ClothingCategory.ACCESSORIES
            ],
            ClothingCategory.BOTTOMS: [
                ClothingCategory.TOPS, ClothingCategory.SHOES,
                ClothingCategory.OUTERWEAR, ClothingCategory.ACCESSORIES
            ],
            ClothingCategory.DRESSES: [
                ClothingCategory.SHOES, ClothingCategory.OUTERWEAR,
                ClothingCategory.ACCESSORIES
            ],
            ClothingCategory.SHOES: [
                ClothingCategory.TOPS, ClothingCategory.BOTTOMS,
                ClothingCategory.DRESSES, ClothingCategory.OUTERWEAR
            ],
            ClothingCategory.OUTERWEAR: [
                ClothingCategory.TOPS, ClothingCategory.BOTTOMS,
                ClothingCategory.DRESSES, ClothingCategory.SHOES
            ],
            ClothingCategory.ACCESSORIES: [
                ClothingCategory.TOPS, ClothingCategory.BOTTOMS,
                ClothingCategory.DRESSES, ClothingCategory.OUTERWEAR
            ]
        }
    
    def _initialize_style_rules(self) -> Dict[Style, List[Style]]:
        """Initialize style compatibility rules."""
        return {
            Style.CASUAL: [Style.CASUAL, Style.SPORTY],
            Style.FORMAL: [Style.FORMAL, Style.BUSINESS],
            Style.BUSINESS: [Style.BUSINESS, Style.FORMAL],
            Style.SPORTY: [Style.SPORTY, Style.CASUAL],
            Style.VINTAGE: [Style.VINTAGE, Style.CASUAL]
        }
    
    def _initialize_color_rules(self) -> Dict[str, float]:
        """Initialize color harmony scoring rules."""
        return {
            'complementary': 0.9,      # Opposite colors
            'analogous': 0.8,          # Adjacent colors
            'triadic': 0.7,            # Three evenly spaced colors
            'monochromatic': 0.85,     # Same hue, different saturation/brightness
            'neutral': 0.75            # Neutral colors (black, white, gray, beige)
        }
    
    def get_recommendations(self, request: RecommendationRequest,
                          available_items: List[ClothingItem]) -> List[OutfitRecommendation]:
        """Generate outfit recommendations based on rules."""
        recommendations = []
        target_item = request.target_item
        
        # Filter available items based on compatibility
        compatible_items = self._filter_compatible_items(target_item, available_items, request)
        
        # Score and rank items
        scored_items = []
        for item in compatible_items:
            score = self.calculate_compatibility_score(target_item, item)
            scored_items.append((item, score))
        
        # Sort by score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        for item, score in scored_items[:request.max_recommendations]:
            reasoning = self._generate_reasoning(target_item, item, score)
            
            recommendation = OutfitRecommendation(
                target_item=target_item,
                recommended_items=[item],
                confidence_score=score,
                reasoning=reasoning
            )
            recommendations.append(recommendation)
        
        self.logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _filter_compatible_items(self, target_item: ClothingItem,
                                available_items: List[ClothingItem],
                                request: RecommendationRequest) -> List[ClothingItem]:
        """Filter items based on basic compatibility rules."""
        compatible = []
        
        for item in available_items:
            # Skip the target item itself
            if item.id == target_item.id:
                continue
            
            # Check category compatibility
            if item.category not in self.category_compatibility.get(target_item.category, []):
                continue
            
            # Check excluded categories
            if item.category in request.exclude_categories:
                continue
            
            # Check style preference
            if request.style_preference and item.style != request.style_preference:
                continue
            
            # Check season preference
            if request.season_preference and item.season != request.season_preference:
                continue
            
            compatible.append(item)
        
        return compatible
    
    def calculate_compatibility_score(self, item1: ClothingItem, item2: ClothingItem) -> float:
        """Calculate compatibility score between two items."""
        scores = []
        
        # Category compatibility score
        category_score = self._calculate_category_score(item1, item2)
        scores.append(category_score * 0.3)  # 30% weight
        
        # Style compatibility score
        style_score = self._calculate_style_score(item1, item2)
        scores.append(style_score * 0.25)  # 25% weight
        
        # Color compatibility score
        color_score = self._calculate_color_score(item1, item2)
        scores.append(color_score * 0.35)  # 35% weight
        
        # Feature similarity score (if features available)
        if item1.features is not None and item2.features is not None:
            feature_score = self._calculate_feature_score(item1, item2)
            scores.append(feature_score * 0.1)  # 10% weight
        
        # Calculate weighted average
        total_score = sum(scores) / len(scores) if scores else 0.0
        return min(max(total_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_category_score(self, item1: ClothingItem, item2: ClothingItem) -> float:
        """Calculate category compatibility score."""
        compatible_categories = self.category_compatibility.get(item1.category, [])
        return 1.0 if item2.category in compatible_categories else 0.0
    
    def _calculate_style_score(self, item1: ClothingItem, item2: ClothingItem) -> float:
        """Calculate style compatibility score."""
        if item1.style is None or item2.style is None:
            return 0.5  # Neutral score for unknown styles
        
        compatible_styles = self.style_compatibility.get(item1.style, [])
        if item2.style in compatible_styles:
            return 1.0 if item1.style == item2.style else 0.8
        return 0.3
    
    def _calculate_color_score(self, item1: ClothingItem, item2: ClothingItem) -> float:
        """Calculate color harmony score between two items."""
        if not item1.colors or not item2.colors:
            return 0.5  # Neutral score for missing color info
        
        best_score = 0.0
        
        # Compare all color combinations
        for color1 in item1.colors:
            for color2 in item2.colors:
                harmony_score = self._calculate_color_harmony(color1, color2)
                best_score = max(best_score, harmony_score)
        
        return best_score
    
    def _calculate_color_harmony(self, color1: Color, color2: Color) -> float:
        """Calculate harmony score between two colors."""
        # Convert RGB to HSV for better color analysis
        h1, s1, v1 = colorsys.rgb_to_hsv(color1.r/255, color1.g/255, color1.b/255)
        h2, s2, v2 = colorsys.rgb_to_hsv(color2.r/255, color2.g/255, color2.b/255)
        
        # Check for neutral colors
        if self._is_neutral_color(color1) or self._is_neutral_color(color2):
            return self.color_harmony_rules['neutral']
        
        # Calculate hue difference
        hue_diff = abs(h1 - h2)
        hue_diff = min(hue_diff, 1.0 - hue_diff)  # Handle circular nature of hue
        
        # Monochromatic (same hue, different saturation/value)
        if hue_diff < 0.05:
            return self.color_harmony_rules['monochromatic']
        
        # Complementary colors (opposite on color wheel)
        if 0.45 <= hue_diff <= 0.55:
            return self.color_harmony_rules['complementary']
        
        # Analogous colors (adjacent on color wheel)
        if hue_diff <= 0.15:
            return self.color_harmony_rules['analogous']
        
        # Triadic colors (120 degrees apart)
        if 0.3 <= hue_diff <= 0.37:
            return self.color_harmony_rules['triadic']
        
        # Default compatibility based on saturation and value similarity
        sat_diff = abs(s1 - s2)
        val_diff = abs(v1 - v2)
        similarity = 1.0 - (sat_diff + val_diff) / 2.0
        
        return similarity * 0.6  # Base compatibility
    
    def _is_neutral_color(self, color: Color) -> bool:
        """Check if a color is neutral (black, white, gray, beige)."""
        r, g, b = color.r, color.g, color.b
        
        # Check for grayscale (r ≈ g ≈ b)
        if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            return True
        
        # Check for beige/tan colors
        if (r > 180 and g > 150 and b > 100 and 
            r - g < 50 and r - b < 80):
            return True
        
        return False
    
    def _calculate_feature_score(self, item1: ClothingItem, item2: ClothingItem) -> float:
        """Calculate feature similarity score using cosine similarity."""
        try:
            similarity = cosine_similarity(
                item1.features.reshape(1, -1),
                item2.features.reshape(1, -1)
            )[0][0]
            
            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1.0) / 2.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating feature similarity: {e}")
            return 0.5
    
    def _generate_reasoning(self, target_item: ClothingItem, 
                          recommended_item: ClothingItem, score: float) -> Dict[str, str]:
        """Generate human-readable reasoning for the recommendation."""
        reasoning = {}
        
        # Category reasoning
        reasoning['category'] = f"{recommended_item.category.value} pairs well with {target_item.category.value}"
        
        # Style reasoning
        if target_item.style and recommended_item.style:
            if target_item.style == recommended_item.style:
                reasoning['style'] = f"Both items have {target_item.style.value} style"
            else:
                reasoning['style'] = f"{recommended_item.style.value} complements {target_item.style.value}"
        
        # Color reasoning
        if target_item.colors and recommended_item.colors:
            color_score = self._calculate_color_score(target_item, recommended_item)
            if color_score > 0.8:
                reasoning['color'] = "Colors create excellent harmony"
            elif color_score > 0.6:
                reasoning['color'] = "Colors work well together"
            else:
                reasoning['color'] = "Colors provide interesting contrast"
        
        reasoning['confidence'] = f"Overall compatibility: {score:.1%}"
        
        return reasoning

class MLRecommendationEngine(RecommendationEngine):
    """Machine learning-based recommendation engine."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_compatibility_score(self, item1: ClothingItem, item2: ClothingItem) -> float:
        """Calculate ML-based compatibility score."""
        if item1.features is None or item2.features is None:
            return 0.0
        
        emb1 = item1.features
        emb2 = item2.features

        score = cosine_similarity(emb1, emb2)

        return float(score)

    def get_recommendations(self, request: RecommendationRequest,
                            available_items: List[ClothingItem]) -> List[OutfitRecommendation]:
        """Rank available items by similarity to the target item."""
        target_item = request.target_item

        scored_items = []
        for item in available_items:
            if item.id == target_item.id:
                continue  # skip the same item
            if item.features is None:
                continue  # skip items without features
            score = self.calculate_compatibility_score(target_item, item)
            scored_items.append((item, score))

        # Sort descending by score
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Take top-N
        recommendations = [
            OutfitRecommendation(
                target_item=target_item,
                recommended_items=[item],
                confidence_score=score,
            )
            for item, score in scored_items[:request.max_recommendations]
        ]
        return recommendations
      