import json
import pickle
import numpy as np
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
        self.embeddings_dir = self.data_dir / "embeddings"
        self.metadata_dir = self.data_dir / "metadata"
        
        self.logger = get_logger(__name__)
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Create necessary directories and files."""
        self.data_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Initialized file storage at {self.data_dir}")
    
    def save_item(self, item: ClothingItem) -> None:
        """Save a clothing item to storage."""
        try:
            # Save item metadata as JSON
            item_file = self.metadata_dir / f"{item.id}.json"
            item_data = self._item_to_dict(item)
            with open(item_file, 'w') as f:
                json.dump(item_data, f, indent=2)
            
            # Save embeddings separately if available
            if item.embedding is not None:
                embedding_file = self.embeddings_dir / f"{item.id}.npy"
                if not embedding_file.exists():
                    np.save(embedding_file, item.embeddings)

            self.logger.debug(f"Saved item: {item.id}")
        except Exception as e:
            self.logger.error(f"Error saving item {item.id}: {str(e)}")
            raise
    
    def load_item(self, item_id: str) -> Optional[ClothingItem]:
        """Load a clothing item by ID."""
        try:
            item_file = self.metadata_dir / f"{item.id}.json"
            if not item_file.exists():
                return None
            
            with open(item_file, 'r') as f:
                item_data = json.load(f)
            
            item = self._dict_to_item(item_data)
            
            embedding_file = self.embeddings_dir / f"{item_id}.npy"
            if embedding_file.exists():
                item.embedding = np.load(embedding_file)
            
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
            'metadata': item.metadata,
            'has_embedding': item.embedding is not None
        }
    
    def _dict_to_item(self, data: Dict[str, Any]) -> ClothingItem:
        """Convert dictionary to ClothingItem."""
        return ClothingItem(
            id=data['id'],
            image_path=Path(data['image_path']),
            category=ClothingCategory(data['category']),
            metadata=data.get('metadata', {}),
            embedding=None  # Loaded separately
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        metadata = self._load_metadata()
        
        stats = {
            'total_items': len(metadata),
            'categories': {},
        }
        
        for item_data in metadata.values():
            # Category stats
            category = item_data.get('category')
            if category:
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
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
                # elif key == 'style' and item.style != value:
                #     match = False
                #     break
                # elif key == 'season' and item.season != value:
                #     match = False
                #     break
            
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