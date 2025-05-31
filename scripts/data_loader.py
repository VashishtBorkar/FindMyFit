import pandas as pd
import json
import os


def parse_all_images(images_dir):
    image_paths = []

    for file in os.scandir(images_dir):
        if file.is_file():
            image_paths.append(file.path)

    return image_paths

def parse_segmented_images(segments_dir):
    segmented_dict = {}
    for file in os.scandir(segments_dir):
        if file.is_file():
            original_name = file.name.replace("_segm.png", ".jpg")
            segmented_dict[original_name] = file.path
    
    return segmented_dict

"""
  0. sleeve length: 0 sleeveless, 1 short-sleeve, 2 medium-sleeve, 3 long-sleeve, 4 not long-sleeve, 5 NA
  1. lower clothing length: 0 three-point, 1 medium short, 2 three-quarter, 3 long, 4 NA
  2. socks: 0 no, 1 socks, 2 leggings, 3 NA
  3. hat: 0 no, 1 yes, 2 NA
  4. glasses: 0 no, 1 eyeglasses, 2 sunglasses, 3 have a glasses in hand or clothes, 4 NA
  5. neckwear: 0 no, 1 yes, 2 NA
  6. wrist wearing: 0 no, 1 yes, 2 NA
  7. ring: 0 no, 1 yes, 2 NA
  8. waist accessories: 0 no, 1 belt, 2 have a clothing, 3 hidden, 4 NA
  9. neckline: 0 V-shape, 1 square, 2 round, 3 standing, 4 lapel, 5 suspenders, 6 NA
  10. outer clothing a cardigan?: 0 yes, 1 no, 2 NA
  11. upper clothing covering navel: 0 no, 1 yes, 2 NA

  Note: 'NA' means the relevant part is not visible.

   <img_name> <shape_0> <shape_1> ... <shape_11>

Taken from: https://github.com/yumingj/DeepFashion-MultiModal

"""
def parse_shape_annotations(shapes_path):
    annotations_dict = {}
    with open(shapes_path, "r") as shape_annotations:
        for line in shape_annotations.readlines():
            parts = line.strip().split()
            name = parts[0]
            attributes = list(map(int, parts[1:]))
            annotations_dict[name] = attributes

    return annotations_dict
            
def parse_image_captions(captions_path):
    with open(captions_path, "r") as captions_file:
        captions_dict = json.load(captions_file)

    return captions_dict

"""
  0 denim, 1 cotton, 2 leather, 3 furry, 4 knitted, 5 chiffon, 6 other, 7 NA

  Note: 'NA' means the relevant part is not visible.
  
  <img_name> <upper_fabric> <lower_fabric> <outer_fabric>

Taken from: https://github.com/yumingj/DeepFashion-MultiModal

"""
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

"""
  0 floral, 1 graphic, 2 striped, 3 pure color, 4 lattice, 5 other, 6 color block, 7 NA

  Note: 'NA' means the relevant part is not visible.
  
  <img_name> <upper_color> <lower_color> <outer_color>

Taken from: https://github.com/yumingj/DeepFashion-MultiModal

"""
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
    count = 0
    for file in os.scandir(images_dir):
        if (
            file.is_file()
            and file.name in segm_dict
            and file.name in captions_dict 
            and file.name in shapes_dict 
            and file.name in fabrics_dict 
            and file.name in patterns_dict
        ):
            count += 1
            image_paths.append(file.path)
            segm_image_paths.append(segm_dict[file.name])
            
            captions.append(captions_dict[file.name])
            shapes.append(shapes_dict[file.name])
            fabrics.append(fabrics_dict[file.name])
            patterns.append(patterns_dict[file.name])

    df = pd.DataFrame({"image_paths" : image_paths, "segm_image_paths": segm_image_paths, "captions": captions})

    shape_columns = [
        "sleeve_length", "lower_clothing_length", "socks", "hat", "glasses",
        "neckwear", "wrist_wearing", "ring", "waist_accessories", "neckline",
        "cardigan", "covers_navel"
    ]
    shape_df = pd.DataFrame(shapes, columns=shape_columns)

    fabric_columns = ["upper_fabric", "lower_fabric", "outer_fabric"]
    fabric_df = pd.DataFrame(fabrics, columns=fabric_columns)

    pattern_columns = ["upper_pattern", "lower_pattern", "outer_pattern"]
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

    #segment_paths = list(parse_segmented_images(segments_dir))
    df = load_raw_data(images_dir, segments_dir, captions_path, shapes_path, fabrics_path, patterns_path)
    print(df.shape)
    print(df.head())


    
