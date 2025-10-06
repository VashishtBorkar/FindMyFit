import sys
import os
from dotenv import load_dotenv
from pathlib import Path
from src.fashion_matcher.clothing_recommender import ClothingRecommender
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
test_data = Path(os.getenv("TEST_DATA_DIR"))
images_dir = Path(os.getenv("IMAGES_DIR"))
clip_embeddings_dir = Path(os.getenv("CLIP_EMBEDDINGS_DIR"))


def main():
    recommender = ClothingRecommender(
        recommendation_engine_type='cosine',
        embeddings_dir=clip_embeddings_dir,
        images_dir=images_dir
    )

    recommendations = recommender.get_recommendations(
        image_path=Path(test_data) / "black_sweatshirt.png",
        target_category='outwear',
        match_categories=['pants'],
        max_recommendations=5,
    )
    
    if not recommendations:
        print("No recommendations returned. :(")
        return 

    print(f"Total Recommendations: {len(recommendations)}")
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"Image: {rec.recommended_item.image_path}, Score: {rec.confidence_score:.4f}")
if __name__ == "__main__":
    main()
