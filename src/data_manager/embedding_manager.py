import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.utils.logging import get_logger
from collections import defaultdict

class EmbeddingManager:

    def __init__(self, embeddings_dir: str, pickle_path: Optional[str] = None):
        self.embeddings_dir = Path(embeddings_dir)
        self.pickle_path = Path(pickle_path) if pickle_path else self.embeddings_dir / "embeddings_cache.pkl"
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.logger = get_logger(__name__)

    def load_embeddings(self, force_reload: bool = False): # add typed return
        """Load embeddings either from pickle or from disk."""
        if self.pickle_path.exists() and not force_reload:
            self.logger.info(f"Loading embeddings from cache: {self.pickle_path}")
            with open(self.pickle_path, "rb") as f:
                self.embeddings = pickle.load(f)
            return self.embeddings, self._build_category_index()

        self.logger.info("Loading embeddings from disk...")
        self.embeddings.clear()

        for category_dir in self.embeddings_dir.iterdir():
            if not category_dir.is_dir():
                continue
            category_name = category_dir.name
            # self.embeddings[category_name] = {}

            for emb_file in category_dir.glob("*.npy"):
                item_id = emb_file.stem
                self.embeddings[item_id] = {
                    "category" : category_name,
                    "embedding" : np.load(emb_file),
                }

            self.logger.info(f"Loaded {len(self.embeddings)} embeddings")

        # Save to pickle
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Saved embeddings to pickle: {self.pickle_path}")

        return self.embeddings, self._build_category_index()

    def _build_category_index(self) -> Dict[str, set]:
        """Build an index of categories to item IDs."""
        category_index = defaultdict(set)
        for item_id, data in self.embeddings.items():
            category_index[data["category"]].add(item_id)
        return category_index

    def get_embedding(self, category: str, item_id: str) -> Optional[np.ndarray]:
        """Retrieve an embedding for a specific item."""
        return self.embeddings.get(item_id)
