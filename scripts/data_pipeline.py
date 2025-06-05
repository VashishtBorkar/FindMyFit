from pathlib import Path
from .data_loader import load_raw_data
from .data_preprocessor import process_raw_data, create_pieces_dataframe, create_embedded_pairs

def load_and_process_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path):
    raw_data = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)
    processed_data = process_raw_data(raw_data)
    
    return processed_data

def build_piece_embeddings(data, cache_dir=Path("embeddings")):
    cache_dir.mkdir(exist_ok=True)  # ensures the folder exists
    return create_pieces_dataframe(data, cache_dir)


def get_training_pairs():
    images_dir = Path(r"D:\Datasets\FindMyFitDatasets\deepfashion\images")
    segments_dir = Path(r"D:\Datasets\FindMyFitDatasets\deepfashion\segm\segm")
    captions_path = Path(r"D:\Datasets\FindMyFitDatasets\deepfashion\captions.json")
    shapes_path = Path(r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\shape\shape_anno_all.txt")
    fabrics_path = Path(r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\fabric_ann.txt")
    patterns_path = Path(r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\pattern_ann.txt")

    # Load and process data from deepfashion files
    raw_data = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)
    data = process_raw_data(raw_data)
    print(f"Loaded {len(data)} outfits")

    # Extract embeddings
    pieces_df = create_pieces_dataframe(data)
    print(f"Extracted {len(pieces_df)} pieces from {len(data)} outfits")
    

    # Create positive negative pairs 
    pairs = create_embedded_pairs(pieces_df)
    print(f"Created {len(pairs)} positive/negative pairs")

    return pairs

if __name__ == "__main__":
    pairs = get_training_pairs()

    print(pairs)
