import itertools
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from src.utils.logging import get_logger

Pair = Tuple[str, str, int]  # (item_id_a, item_id_b, label)

def load_pairs(embeddings: Dict, outfit_file: Path, output_pickle: Path, force_reload: bool = False):
    """Load compatibility pairs"""
    logger = get_logger(__name__)
    if output_pickle.exists() and not force_reload:
        logger.info(f"Loading pairs from {output_pickle}...")
        with open(output_pickle, "rb") as f:
            pairs = pickle.load(f)
        return pairs
    
    pairs = []
    logger.info(f"Creating pairs from {outfit_file}...")
    with open(outfit_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = int(parts[0])
            item_ids = parts[1:]

            for item1, item2 in itertools.permutations(item_ids, 2):
                if item1 in embeddings and item2 in embeddings:
                    pairs.append((item1, item2, label))

    logger.info(f"Generated {len(pairs)} pairs. Saving to {output_pickle}...")
    with open(output_pickle, "wb") as f:
        pickle.dump(pairs, f)
    logger.info("Finished creating compatibility pairs.")
    return pairs

class PairDataset(Dataset):
    """Dataset yielding (embedding_a, embedding_b, label) tensors."""

    def __init__(self, pairs: List[Pair], embeddings: Dict[str, Dict[str, np.ndarray]]):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_id, b_id, label = self.pairs[idx]
        emb_a = torch.tensor(self.embeddings[a_id]["embedding"], dtype=torch.float32)
        emb_b = torch.tensor(self.embeddings[b_id]["embedding"], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return emb_a, emb_b, label
    

