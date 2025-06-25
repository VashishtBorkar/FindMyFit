import pandas as pd
import json
import os

# Create list of all image names in directory
def parse_all_images(images_dir):
    image_paths = [
        file.path for file in os.scandir(images_dir)
        if file.is_file()
    ]
    return image_paths

# Map original files to segment mask 
def parse_segmented_images(segments_dir):
    segmented_dict = {}
    for file in os.scandir(segments_dir):
        if file.is_file():
            original_name = file.name.replace("_segm.png", ".jpg")
            segmented_dict[original_name] = file.path
    
    return segmented_dict


# Parses shape annotations for each image and returns a dict
def parse_shape_annotations(shapes_path):
    annotations_dict = {}
    with open(shapes_path, "r") as shape_annotations:
        for line in shape_annotations.readlines():
            parts = line.strip().split()
            name = parts[0]
            attributes = list(map(int, parts[1:]))
            annotations_dict[name] = attributes

    return annotations_dict

# Loads captions from JSON file
def parse_image_captions(captions_path):
    with open(captions_path, "r") as captions_file:
        captions_dict = json.load(captions_file)

    return captions_dict


# Parses fabric annotations for each image and returns a dict
def parse_fabric_annotations(fabrics_path):
    fabrics_dict = {}
    with open(fabrics_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            name = parts[0]
            fabrics = list(map(int, parts[1:]))
            fabrics_dict[name] = fabrics

    return fabrics_dict


# Parses pattern annotations for each image and returns a dict
def parse_pattern_annotations(patterns_path):
    patterns_dict = {}
    with open(patterns_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            name = parts[0]
            patterns = list(map(int, parts[1:]))
            patterns_dict[name] = patterns

    return patterns_dict


# loads and merges all metadata into single dataframe
def load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path):
    all_images = parse_all_images(images_dir)
    segm_dict = parse_segmented_images(segments_dir)
    captions_dict = parse_image_captions(captions_path)
    shapes_dict = parse_shape_annotations(shapes_path)
    fabrics_dict = parse_fabric_annotations(fabrics_path)
    patterns_dict = parse_pattern_annotations(patterns_path)

    image_paths = []
    segm_image_paths = []
    captions = []
    shapes = []
    fabrics = []
    patterns = []

    print("Scanning images...")

    for file in os.scandir(images_dir):
        if (
            file.is_file()
            and file.name in segm_dict
            and file.name in captions_dict 
            and file.name in shapes_dict 
            and file.name in fabrics_dict 
            and file.name in patterns_dict
        ):
            image_paths.append(file.path)
            segm_image_paths.append(segm_dict[file.name])
            captions.append(captions_dict[file.name])
            shapes.append(shapes_dict[file.name])
            fabrics.append(fabrics_dict[file.name])
            patterns.append(patterns_dict[file.name])

    df = pd.DataFrame({
        "image_paths" : image_paths, 
        "segm_image_paths": segm_image_paths, 
        "captions": captions,
    })

    shape_columns = [
        "sleeve_length", "lower_clothing_length", "socks", "hat", "glasses",
        "neckwear", "wrist_wearing", "ring", "waist_accessories", "neckline",
        "cardigan", "covers_navel"
    ]

    fabric_columns = ["upper_fabric", "lower_fabric", "outer_fabric"]
    pattern_columns = ["upper_pattern", "lower_pattern", "outer_pattern"]

    shape_df = pd.DataFrame(shapes, columns=shape_columns)
    fabric_df = pd.DataFrame(fabrics, columns=fabric_columns)
    pattern_df = pd.DataFrame(patterns, columns=pattern_columns)

    data = pd.concat([df, shape_df, fabric_df, pattern_df], axis=1)

    return data

if __name__ == "__main__":

    images_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\images"
    segments_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\segm\segm"
    captions_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\captions.json"
    shapes_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\shape\shape_anno_all.txt"
    fabrics_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\fabric_ann.txt"
    patterns_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\texture\pattern_ann.txt"

    df = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)
    print(df.head())
    print("\nShape: ", df.shape)
    print("Sample caption:", df['captions'].iloc[0])
