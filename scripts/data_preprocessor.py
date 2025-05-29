import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import clip

from data_loader import load_raw_data

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.squeeze().cpu().numpy()

def preprocess_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path):
    df = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)

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
        df[col] = df[col].replace(na_val, -1)  # Optionally use np.nan

    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

    return df_encoded




images_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\images"
captions_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\captions.json"
annotations_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\shape\shape_anno_all.txt"
fabrics_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\fabric_ann.txt"
patterns_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\pattern_ann.txt"

data = load_data(images_dir, captions_path, annotations_path, fabrics_path, patterns_path)

data["image_embedding"] = data["image_paths"].apply(get_image_embedding)



