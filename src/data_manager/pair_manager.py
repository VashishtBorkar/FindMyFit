import itertools
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from src.data_manager.embedding_manager import EmbeddingManager
from src.utils.logging import get_logger

Pair = Tuple[str, str, int]  # (item_id_a, item_id_b, label)

class PairManager:
    def __init__(self, embeddings, outfit_file: Path, output_pickle: Path):
        self.embeddings = embeddings
        self.outfit_file = outfit_file
        self.output_pickle = output_pickle
        self.logger = get_logger(__name__)
    
    def load_pairs(self, force_reload: bool = False):
        """Load compatibility pairs"""
        if not force_reload and self.output_pickle.exists():
            self.logger.info(f"Loading pairs from {self.output_pickle}...")
            with open(self.output_pickle, "rb") as f:
                pairs = pickle.load(f)
            return pairs
        
        pairs = []
        self.logger.info(f"Creating pairs from {self.outfit_file}...")
        with open(self.outfit_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                label = int(parts[0])
                item_ids = parts[1:]

                for item1, item2 in itertools.permutations(item_ids, 2):
                    if item1 in self.embeddings and item2 in self.embeddings:
                        pairs.append((item1, item2, label))

        self.logger.info(f"Generated {len(pairs)} pairs. Saving to {self.output_pickle}...")
        with open(self.output_pickle, "wb") as f:
            pickle.dump(pairs, f)
        self.logger.info("Finished creating compatibility pairs.")
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
    

