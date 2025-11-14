import itertools
import pickle
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from src.utils.logging import get_logger, setup_logging
from src.data_manager.embedding_manager import load_embeddings 
import os
 
def main():
    # setup_logging(level='DEBUG')
    logger = get_logger(__name__)
    setup_logging(level='INFO')
    
    load_dotenv()
    embeddings_dir = Path("data/clip_embeddings").resolve()
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings, category_index = load_embeddings(embeddings_dir, )

    outfit_file = Path(os.getenv("COMPATIBILE_OUTFITS_FILE", "data/outfits.txt"))
    output_pickle = Path("data/compatibility_pairs.pkl").resolve()

    if not outfit_file.exists():
        raise FileNotFoundError(f"Outfit file not found: {outfit_file}")

    # if output_pickle.exists():
    #     logger.info(f"Output pickle already exists: {output_pickle}. Remove it to recreate pairs.")
    #     return
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

            # Create all unique pairs (combinations) within the outfit
            for item1, item2 in itertools.permutations(item_ids, 2):
                # Only add pairs where both embeddings exist
                if item1 in embeddings and item2 in embeddings:
                    pairs.append((item1, item2, label))

    logger.info(f"Generated {len(pairs)} pairs. Saving to {output_pickle}...")
    with open(output_pickle, "wb") as f:
        pickle.dump(pairs, f)

    logger.info("Finished creating compatibility pairs.")

if __name__ == "__main__":
    main()
    with open("data/compatibility_pairs.pkl", "rb") as f:
        loaded_pairs = pickle.load(f)
    
    print(f"Loaded {len(loaded_pairs)} pairs from pickle.")
    print("Sample pairs:", loaded_pairs[:10])
