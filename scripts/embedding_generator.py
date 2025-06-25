import pandas as pd
import numpy as np
import torch
import clip
from pathlib import Path
from PIL import Image
import logging

from .data_preprocessing import load_raw_data

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
    #'accessories': [16, 17, 18, 19, 22], # ring, wrist wearing, socks, gloves, earrings
}

LABEL_NAMES = {
    1: 'top', 2: 'outer', 3: 'skirt',
    4: 'dress', 5: 'pants', 6: 'leggings',
    7: 'headwear', 8: 'eyeglass', 9: 'neckwear',
    10: 'belt', 11: 'footwear', 12: 'bag',
    16: 'ring', 17: 'wrist wearing', 18: 'socks',
    20: 'necklace', 21: 'rompers', 22: 'earrings', 23: 'tie',
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

logging.basicConfig(filename="embedding_generator.log", level=logging.INFO)

# Converts metadata columns with NA to -1 and one-hot encodes categorical features
def preprocess_metadata(raw_df):
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
        raw_df[col] = raw_df[col].replace(na_val, -1)  # Potentially use np.nan

    encoded_df = pd.get_dummies(raw_df, columns=categorical_cols, prefix=categorical_cols)

    return encoded_df

# Applies soft mask to isolate a piece and then returns embedding
def generate_piece_embedding(image_path, mask_path, label, pixel_threshold=0.01):
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    image = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(mask_path))

    target_mask = (mask == label).astype(np.uint8)
    if np.mean(target_mask) < pixel_threshold:
        return None

    soft_mask = target_mask[..., None] * 255

    guided_image = np.array(image).astype(np.float32)
    guided_image = guided_image * (soft_mask / 255) + 128 * (1 - soft_mask / 255)
    guided_image = Image.fromarray(guided_image.astype(np.uint8))

    with torch.no_grad():
        input_tensor = preprocess(guided_image).unsqueeze(0).to(device)
        embedding = model.encode_image(input_tensor)
        return embedding.squeeze().cpu().numpy()


# Iterate over all outfits and generate embedding for each segmented piece
def extract_piece_embeddings(df, cache_dir="piece_embeddings"):
    cache_dir = Path(cache_dir)
    data = []

    for idx, row in df.iterrows():
        mask = np.array(Image.open(row["segm_image_paths"]))
        unique_labels = np.unique(mask)

        for label in unique_labels:
            piece_type = LABEL_NAMES.get(label)
            if not piece_type:
                continue

            unique_id = f"outfit_{idx}_{piece_type}"
            embedding_path = cache_dir / f"{unique_id}.npy"
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            if not embedding_path.exists():
                embedding = generate_piece_embedding(row["image_paths"], row["segm_image_paths"], label)
                np.save(str(embedding_path), embedding)

            data.append({
                "outfit_id": idx,
                "piece_type": piece_type,
                "embedding_path": embedding_path,
                "meta": row.drop(["image_paths", "segm_image_paths", "captions"]).values.astype(np.float32)
            })

    return pd.DataFrame(data)

if __name__ == "__main__":
    from scripts.data_preprocessing import load_raw_data

    # Set paths to your dataset
    images_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\images"
    segments_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\segm\segm"
    captions_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\captions.json"
    shapes_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\shape\shape_anno_all.txt"
    fabrics_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\fabric_ann.txt"
    patterns_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\pattern_ann.txt"

    # Load and preprocess data
    print("Loading raw dataset...")
    df = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)

    # One-hot encode and clean metadata
    df = preprocess_metadata(df)

    # Small subset for testing
    subset = df.head(5)

    print("Extracting piece embeddings...")
    pieces_df = extract_piece_embeddings(subset, cache_dir="test_piece_embeddings")

    # Output results
    print(f"\nExtracted {len(pieces_df)} clothing pieces")
    print(pieces_df.head())

    # Verify embedding files exist
    missing = pieces_df[~pieces_df['embedding_path'].apply(lambda p: Path(p).exists())]
    if not missing.empty:
        print("\nSome embeddings were not saved correctly:")
        print(missing[['outfit_id', 'piece_type', 'embedding_path']])
    else:
        print("\nAll embedding files found.")

        