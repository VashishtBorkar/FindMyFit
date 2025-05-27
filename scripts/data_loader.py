import pandas as pd
import json
import os

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
"""
NUM_ATTRIBUTES = 12

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

def load_data(images_dir, captions_path, annotations_path):
    captions_dict = parse_image_captions(captions_path)
    annotations_dict = parse_shape_annotations(annotations_path)

    image_paths = []
    captions = []
    annotations = []

    for file in os.scandir(images_dir):
        if file.is_file() and file.name[-len("full.jpg"):] == "full.jpg": # file is a full outfit image
            if file.name in captions_dict and file.name in annotations_dict:
                captions.append(captions_dict[file.name])
                image_paths.append(file.path)
                annotations.append(annotations_dict[file.name])

    data = pd.DataFrame({"image_paths" : image_paths, "captions": captions})

    annotation_names = [
        "sleeve_length",
        "lower_clothing_length",
        "socks",
        "hat",
        "glasses",
        "neckwear",
        "wrist_wearing",
        "ring",
        "waist_accessories",
        "neckline",
        "cardigan",
        "covers_navel"
    ]

    attributes_df = pd.DataFrame(annotations, columns=annotation_names)

    data = pd.concat([data, attributes_df], axis=1)
    return data

if __name__ == "__main__":

    images_dir = r"D:\Datasets\FindMyFitDatasets\deepfashion\images"
    captions_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\captions.json"
    annotations_path = r"D:\Datasets\FindMyFitDatasets\deepfashion\labels\labels\shape\shape_anno_all.txt"

    df = load_data(images_dir, captions_path, annotations_path)

    print(df.head())
    