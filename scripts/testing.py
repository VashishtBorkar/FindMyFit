import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from src.fashion_matcher.clothing_recommender import ClothingRecommender
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
test_data = Path(os.getenv("TEST_DATA_DIR"))
images_dir = Path(os.getenv("IMAGES_DIR"))
clip_embeddings_dir = Path(os.getenv("CLIP_EMBEDDINGS_DIR"))
metric_embeddings_dir = Path(os.getenv("METRIC_EMBEDDINGS_DIR"))


def main():
    max_recommendations = 10

    recommender = ClothingRecommender(
        recommendation_engine_type='metric',
        embeddings_dir=metric_embeddings_dir,
        images_dir=images_dir
    )

    target_image_path = Path(test_data) / "black_sweatpants.png"

    recommendations = recommender.get_recommendations(
        image_path=target_image_path,
        target_category='top',
        match_categories=['top'],
        max_recommendations=max_recommendations,
    )
    
    if not recommendations:
        print("No recommendations returned.")
        return 
    
    print(f"Total Recommendations: {len(recommendations)}")
    print("\nRecommendations (Image Paths):")
    print("Target Image: ", target_image_path)
    for rec in recommendations:
        print(f"Image: {rec.recommended_item.image_path}, Score: {rec.confidence_score:.4f}")

    num_images = max_recommendations + 1  # target + recommendations
    plt.figure(figsize=(3 * num_images, 4))

    # --- Show target item ---
    plt.subplot(1, num_images, 1)
    target_img = Image.open(target_image_path)
    plt.imshow(target_img)
    plt.title("Target Item", fontsize=12)
    plt.axis("off")

    # --- Show recommendations ---
    for i, rec in enumerate(recommendations[:max_recommendations], start=2):
        rec_path = Path(str(rec.recommended_item.image_path) + ".jpg")
        if not rec_path.exists():
            print(f"Image Not Found: {rec_path}")
            continue
        plt.subplot(1, num_images, i)
        rec_img = Image.open(rec_path)
        plt.imshow(rec_img)
        plt.title(f"Score: {rec.confidence_score:.2f}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
