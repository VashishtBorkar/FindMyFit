import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.utils.logging import get_logger

class EmbeddingManager:
    """Handles loading and caching of precomputed embeddings."""

    def __init__(self, embeddings_dir: str, pickle_path: Optional[str] = None):
        self.embeddings_dir = Path(embeddings_dir)
        self.pickle_path = Path(pickle_path) if pickle_path else self.embeddings_dir / "embeddings_cache.pkl"
        self.clip_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.logger = get_logger(__name__)

    def load_embeddings(self, force_reload: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """Load embeddings either from pickle or from disk."""
        if self.pickle_path.exists() and not force_reload:
            self.logger.info(f"Loading embeddings from cache: {self.pickle_path}")
            with open(self.pickle_path, "rb") as f:
                self.clip_embeddings = pickle.load(f)
            return self.clip_embeddings

        self.logger.info("Loading embeddings from disk...")
        self.clip_embeddings.clear()

        for category_dir in self.embeddings_dir.iterdir():
            if not category_dir.is_dir():
                continue
            category_name = category_dir.name
            self.clip_embeddings[category_name] = {}

            for emb_file in category_dir.glob("*.npy"):
                item_id = emb_file.stem
                self.clip_embeddings[category_name][item_id] = np.load(emb_file)

            self.logger.info(f"Loaded {len(self.clip_embeddings[category_name])} embeddings for '{category_name}'")

        # Save to pickle
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.clip_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Saved embeddings to pickle: {self.pickle_path}")

        return self.clip_embeddings

    def get_embedding(self, category: str, item_id: str) -> Optional[np.ndarray]:
        """Retrieve an embedding for a specific item."""
        return self.clip_embeddings.get(category, {}).get(item_id)
