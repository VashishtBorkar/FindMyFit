import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from src.utils.logging import get_logger
from collections import defaultdict

def load_embeddings(
    embeddings_dir: str | Path,
    pickle_path: Optional[str | Path] = None,
    force_reload: bool = False
):    
    """Load embeddings either from pickle or from disk."""
    logger = get_logger(__name__)
    embeddings_dir = Path(embeddings_dir)
    pickle_path = Path(pickle_path) if pickle_path else embeddings_dir / "embeddings_cache.pkl"

    if pickle_path.exists() and not force_reload:
        logger.info(f"Loading embeddings from cache: {pickle_path}")
        with open(pickle_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings, _build_category_index(embeddings)
    
    embeddings = {}
    logger.info("Loading embeddings from disk...")

    for category_dir in embeddings_dir.iterdir():
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name
        for emb_file in category_dir.glob("*.npy"):
            item_id = emb_file.stem
            embeddings[item_id] = {
                "category" : category_name,
                "embedding" : np.load(emb_file),
            }

        logger.info(f"Loaded {len(embeddings)} embeddings")

    # Save to pickle
    with open(pickle_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved embeddings to pickle: {pickle_path}")

    return embeddings, _build_category_index(embeddings)

def _build_category_index(embeddings) -> Dict[str, set]:
    """Build an index of categories to item IDs."""
    category_index = defaultdict(set)
    for item_id, data in embeddings.items():
        category_index[data["category"]].add(item_id)
    return category_index
