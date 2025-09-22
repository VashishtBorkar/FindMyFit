import sys
import os
from dotenv import load_dotenv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
load_dotenv()
test_data = os.getenv("TEST_DATA_DIR")
from fashion_matcher import FashionMatcher
from fashion_matcher.core.models import ClothingCategory

def main():
    fm = FashionMatcher(
        data_dir=Path("data"), 
        feature_extractor_type="simple",
        recommendation_engine_type="ml", 
        storage_type="memory"
    )

    image_dir = Path(test_data)
    print(image_dir)
    items = fm.batch_process_images(
        image_directory=image_dir,
        default_category=ClothingCategory.TOPS
    )
    
    if not items:
        print("No items processed.")
        return

    target_item = items[0]
    print(f"Target item: {target_item.image_path.name} ({target_item.category})")

    # Currently Cosine similarity of two images
    recs = fm.get_recommendations(
        target_item_id=target_item.id,
        max_recommendations=3
    )
    
    if not recs:
        print("No recommendations returned. :(")
        return 
    
    print("\nRecommendations:")
    for rec in recs:
        rec_items = ", ".join([r.image_path.name for r in rec.recommended_items])
        print(f" - Items: {rec_items}, Score: {rec.confidence_score:.3f}")

if __name__ == "__main__":
    main()
