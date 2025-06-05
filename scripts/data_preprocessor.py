import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import clip
from itertools import combinations
import random
from pathlib import Path

from .data_loader import load_raw_data

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

CLOTHING_PIECES = {
    'top': [1],           # top
    'outer': [2],         # outer layer (jackets, coats)
    'bottom': [3, 5, 6],  # skirt, pants, leggings
    'dress': [4, 21],     # dress, rompers
    'headwear': [7],      # headwear
    'eyewear': [8],       # eyeglass
    'neckwear': [9, 20, 23], # neckwear, necklace, tie
    'belt': [10],         # belt
    'footwear': [11],     # footwear
    'bag': [12],          # bag
    'accessories': [16, 17, 18, 19, 22], # ring, wrist wearing, socks, gloves, earrings
}

def process_raw_data(raw_df):
    categorical_cols = [
        "socks", "hat", "glasses", "neckwear", "wrist_wearing", "ring",
        "waist_accessories", "neckline", "cardigan", "covers_navel",
        "upper_fabric", "lower_fabric", "outer_fabric",
        "upper_pattern", "lower_pattern", "outer_pattern"
    ]

    na_labels = {
        "glasses": 4, "waist_accessories": 4, "neckline": 6, "cardigan": 2,
        "covers_navel": 2, "upper_fabric": 7, "lower_fabric": 7, "outer_fabric": 7, 
        "upper_pattern": 7, "lower_pattern": 7,  "outer_pattern": 7,
    }

    for col, na_val in na_labels.items():
        raw_df[col] = raw_df[col].replace(na_val, -1)  # Optionally use np.nan

    df_encoded = pd.get_dummies(raw_df, columns=categorical_cols, prefix=categorical_cols)

    return df_encoded

def get_all_piece_embeddings(image_path, mask_path, clothing_piece_dict, outfit_idx, cache_dir=Path("embeddings"), pixel_threshold=0.01):
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    mask_array = np.array(mask)
    unique_labels = np.unique(mask_array)
    print(f"Outfit {outfit_idx} has unique labels: {unique_labels}")

    embeddings = {}
    for piece_type, label_ids in clothing_piece_dict.items():
        try:
            target_mask = np.isin(mask_array, label_ids).astype(np.uint8)
            if np.mean(target_mask) < pixel_threshold:
                continue

            # Create masked image with background blended
            mask_normalized = target_mask[..., None] * 255
            guided_image = np.array(image).astype(np.float32)
            guided_image = guided_image * (mask_normalized / 255) + 128 * (1 - mask_normalized / 255)
            guided_image = Image.fromarray(guided_image.astype(np.uint8))

            # File path to save embedding
            unique_id = f"outfit_{outfit_idx}_{piece_type}"
            embedding_path = cache_dir / f"{unique_id}.npy"
            embedding_path.parent.mkdir(parents=True, exist_ok=True)

            if embedding_path.exists():
                embedding_np = np.load(embedding_path)
            else:
                # Preprocess and get CLIP embedding
                with torch.no_grad():
                    preprocessed = preprocess(guided_image).unsqueeze(0).to(device)
                    embedding = model.encode_image(preprocessed)
                    embedding_np = embedding.squeeze().cpu().numpy()
                    np.save(str(embedding_path), embedding_np)
                    del preprocessed, embedding
                    torch.cuda.empty_cache()


            embeddings[piece_type] = embedding_np

        except Exception as e:
            print(f"Error embedding piece {piece_type} in outfit {outfit_idx}: {e}")
    
    return embeddings

def create_pieces_dataframe(data_df, cache_dir=Path("embeddings")):
    pieces_dict = {
        'outfit_id': [],
        'piece_type': [],
        'piece_id': [],
        'embedding': [],
        'image_path': [],
        'mask_path': []
    }

    print("Extracting individual clothing pieces...")
    for idx, row in data_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing outfit {idx}/{len(data_df)}")

        try:
            pieces_embeddings = get_all_piece_embeddings(
                row["image_paths"],
                row["segm_image_paths"],
                CLOTHING_PIECES,
                outfit_idx=idx,
                cache_dir=cache_dir
            )

            for piece_type, embedding in pieces_embeddings.items():
                pieces_dict['outfit_id'].append(idx)
                pieces_dict['piece_type'].append(piece_type)
                pieces_dict['piece_id'].append(f"outfit_{idx}_{piece_type}")
                pieces_dict['embedding'].append(embedding)
                pieces_dict['image_path'].append(row["image_paths"])
                pieces_dict['mask_path'].append(row["segm_image_paths"])

        except Exception as e:
            print(f"Error processing outfit {idx}: {str(e)}")

    pieces_df = pd.DataFrame(pieces_dict)

    print(f"\nExtracted {len(pieces_df)} individual clothing pieces from {data_df.shape[0]} outfits")
    print(f"Piece type distribution:")
    print(pieces_df['piece_type'].value_counts())

    return pieces_df

def create_embedded_pairs(df, num_negative_per_positive=1, seed=42):
    random.seed(seed)
    pairs = []

    # Group all items by outfit
    outfit_groups = df.groupby("outfit_id")

    outfit_ids = df['outfit_id'].unique().tolist()

    for outfit_id, group in outfit_groups:
        pieces = group.to_dict(orient='records')

        # Pairs from same outfit
        for item1, item2 in combinations(pieces, 2):
            pairs.append({
                "embedding_1": item1["embedding"],
                "embedding_2": item2["embedding"],
                "label": 1
            })

            # Pairs from different outfits
            for _ in range(num_negative_per_positive):
                tries = 0
                max_tries = 10  # To avoid infinite loop

                while tries < max_tries:
                    neg_outfit_id = random.choice([oid for oid in outfit_ids if oid != outfit_id])
                    neg_group = df[df['outfit_id'] == neg_outfit_id]
                    neg_item = neg_group.sample(1).iloc[0]

                    # Skip if it's the same piece type (optional) or bad data
                    if (
                        item1['piece_type'] == neg_item['piece_type'] or
                        not isinstance(neg_item['embedding'], np.ndarray)
                    ):
                        tries += 1
                        continue

                    # Valid negative, break out of loop
                    pairs.append({
                        "embedding_1": item1["embedding"],
                        "embedding_2": neg_item["embedding"],
                        "label": 0
                    })
                    break
    random.shuffle(pairs)
    return pairs


"""
0: 'background'	1: 'top'	2: 'outer'	3: 'skirt'
4: 'dress'	5: 'pants'	6: 'leggings'	7: 'headwear'
8: 'eyeglass'	9: 'neckwear'	10: 'belt'	11: 'footwear'
12: 'bag'	13: 'hair'	14: 'face'	15: 'skin'
16: 'ring'	17: 'wrist wearing'	18: 'socks'	19: 'gloves'
20: 'necklace'	21: 'rompers'	22: 'earrings'	23: 'tie'
"""

if __name__ == "__main__":
    # paths to datasets
    images_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\images"
    segments_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\segm\segm"
    captions_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\captions.json"
    shapes_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\shape\shape_anno_all.txt"
    fabrics_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\fabric_ann.txt"
    patterns_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\pattern_ann.txt"

    # Load and process data
    raw_data = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)
    data = process_raw_data(raw_data)

    # Use only the first 10 outfits for testing
    subset = data.head(10)

    # Extract embeddings
    pieces_df = create_pieces_dataframe(subset)

    # Create positive negative pairs 
    pairs = create_embedded_pairs(pieces_df)

    # Show preview
    print("Total Data Shape: ", data.shape)
    print("Preview: ", subset.head())
    print("Pieces Shape: ", pieces_df.shape)
    print("Preview: ", pieces_df.head())
    print("Length of pairs: ", len(pairs))

    