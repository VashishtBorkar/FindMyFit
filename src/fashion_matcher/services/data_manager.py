import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from ..core.interfaces import DataStorage
from ..core.models import ClothingItem, ClothingCategory, Style, Season, Color
from ..utils.logging import get_logger


class FileBasedDataStorage(DataStorage):
    """File-based data storage implementation for the prototype."""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.items_dir = self.data_dir / "items"
        self.features_dir = self.data_dir / "features"
        self.metadata_file = self.data_dir / "metadata.json"
        
        self.logger = get_logger(__name__)
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Create necessary directories and files."""
        self.data_dir.mkdir(exist_ok=True)
        self.items_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)
        
        if not self.metadata_file.exists():
            self._save_metadata({})
        
        self.logger.info(f"Initialized file storage at {self.data_dir}")
    
    def save_item(self, item: ClothingItem) -> None:
        """Save a clothing item to storage."""
        try:
            # Save item metadata as JSON
            item_data = self._item_to_dict(item)
            item_file = self.items_dir / f"{item.id}.json"
            
            with open(item_file, 'w') as f:
                json.dump(item_data, f, indent=2)
            
            # Save features separately if available
            if item.features is not None:
                feature_file = self.features_dir / f"{item.id}.pkl"
                with open(feature_file, 'wb') as f:
                    pickle.dump(item.features, f)
            
            # Update metadata index
            self._update_metadata_index(item)
            
            self.logger.debug(f"Saved item: {item.id}")
            
        except Exception as e:
            self.logger.error(f"Error saving item {item.id}: {str(e)}")
            raise
    
    def load_item(self, item_id: str) -> Optional[ClothingItem]:
        """Load a clothing item by ID."""
        try:
            item_file = self.items_dir / f"{item_id}.json"
            
            if not item_file.exists():
                return None
            
            # Load item data
            with open(item_file, 'r') as f:
                item_data = json.load(f)
            
            item = self._dict_to_item(item_data)
            
            # Load features if available
            feature_file = self.features_dir / f"{item_id}.pkl"
            if feature_file.exists():
                with open(feature_file, 'rb') as f:
                    item.features = pickle.load(f)
            
            return item
            
        except Exception as e:
            self.logger.error(f"Error loading item {item_id}: {str(e)}")
            return None
    
    def get_all_items(self) -> List[ClothingItem]:
        """Get all stored clothing items."""
        items = []
        
        for item_file in self.items_dir.glob("*.json"):
            item_id = item_file.stem
            item = self.load_item(item_id)
            if item:
                items.append(item)
        
        self.logger.info(f"Loaded {len(items)} items from storage")
        return items
    
    def search_items(self, **filters) -> List[ClothingItem]:
        """Search items with filters."""
        all_items = self.get_all_items()
        filtered_items = []
        
        for item in all_items:
            if self._matches_filters(item, filters):
                filtered_items.append(item)
        
        self.logger.debug(f"Found {len(filtered_items)} items matching filters")
        return filtered_items
    
    def _matches_filters(self, item: ClothingItem, filters: Dict[str, Any]) -> bool:
        """Check if an item matches the given filters."""
        for key, value in filters.items():
            if key == 'category' and item.category != value:
                return False
            elif key == 'style' and item.style != value:
                return False
            elif key == 'season' and item.season != value:
                return False
            elif key == 'has_features' and bool(item.features is not None) != value:
                return False
        
        return True
    
    def _item_to_dict(self, item: ClothingItem) -> Dict[str, Any]:
        """Convert ClothingItem to dictionary for JSON serialization."""
        return {
            'id': item.id,
            'image_path': str(item.image_path),
            'category': item.category.value,
            'style': item.style.value if item.style else None,
            'season': item.season.value if item.season else None,
            'colors': [
                {'r': c.r, 'g': c.g, 'b': c.b, 'name': c.name}
                for c in item.colors
            ],
            'metadata': item.metadata,
            'has_features': item.features is not None
        }
    
    def _dict_to_item(self, data: Dict[str, Any]) -> ClothingItem:
        """Convert dictionary to ClothingItem."""
        colors = [
            Color(r=c['r'], g=c['g'], b=c['b'], name=c.get('name'))
            for c in data.get('colors', [])
        ]
        
        return ClothingItem(
            id=data['id'],
            image_path=Path(data['image_path']),
            category=ClothingCategory(data['category']),
            style=Style(data['style']) if data.get('style') else None,
            season=Season(data['season']) if data.get('season') else None,
            colors=colors,
            metadata=data.get('metadata', {})
        )
    
    def _update_metadata_index(self, item: ClothingItem):
        """Update the metadata index with item information."""
        try:
            metadata = self._load_metadata()
            
            metadata[item.id] = {
                'category': item.category.value,
                'style': item.style.value if item.style else None,
                'season': item.season.value if item.season else None,
                'has_features': item.features is not None,
                'colors_count': len(item.colors)
            }
            
            self._save_metadata(metadata)
            
        except Exception as e:
            self.logger.warning(f"Error updating metadata index: {str(e)}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata index."""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata index."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        metadata = self._load_metadata()
        
        stats = {
            'total_items': len(metadata),
            'categories': {},
            'styles': {},
            'seasons': {},
            'items_with_features': 0
        }
        
        for item_data in metadata.values():
            # Category stats
            category = item_data.get('category')
            if category:
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Style stats
            style = item_data.get('style')
            if style:
                stats['styles'][style] = stats['styles'].get(style, 0) + 1
            
            # Season stats
            season = item_data.get('season')
            if season:
                stats['seasons'][season] = stats['seasons'].get(season, 0) + 1
            
            # Features stats
            if item_data.get('has_features'):
                stats['items_with_features'] += 1
        
        return stats


class InMemoryDataStorage(DataStorage):
    """In-memory data storage for testing and development."""
    
    def __init__(self):
        self.items: Dict[str, ClothingItem] = {}
        self.logger = get_logger(__name__)
    
    def save_item(self, item: ClothingItem) -> None:
        """Save item in memory."""
        self.items[item.id] = item
        self.logger.debug(f"Saved item in memory: {item.id}")
    
    def load_item(self, item_id: str) -> Optional[ClothingItem]:
        """Load item from memory."""
        return self.items.get(item_id)
    
    def get_all_items(self) -> List[ClothingItem]:
        """Get all items from memory."""
        return list(self.items.values())
    
    def search_items(self, **filters) -> List[ClothingItem]:
        """Search items in memory."""
        results = []
        
        for item in self.items.values():
            match = True
            
            for key, value in filters.items():
                if key == 'category' and item.category != value:
                    match = False
                    break
                elif key == 'style' and item.style != value:
                    match = False
                    break
                elif key == 'season' and item.season != value:
                    match = False
                    break
            
            if match:
                results.append(item)
        
        return results
    
    def clear(self):
        """Clear all items from memory."""
        self.items.clear()
        self.logger.info("Cleared all items from memory")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        return {
            'total_items': len(self.items),
            'storage_type': 'in_memory'
        }