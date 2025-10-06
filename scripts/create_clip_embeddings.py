from pathlib import Path
import os
from dotenv import load_dotenv
from src.fashion_matcher.services.embedding_generator import CLIPEmbeddingGenerator
from src.fashion_matcher.utils.logging import get_logger, setup_logging  # if you have a logger helper


def main():
    # setup_logging(level='DEBUG')
    logger = get_logger(__name__)
    
    # Paths
    load_dotenv()
    image_dir = Path(os.getenv("IMAGES_DIR", "../data/images"))
    embeddings_dir = Path("data/clip_embeddings").resolve()
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embeddings_dir}")

    # Initialize embedding generator
    generator = CLIPEmbeddingGenerator(model_name="ViT-B/32")

    # Iterate through all images
    processed_count = 0
    for category_dir in Path(image_dir).iterdir():
        if not category_dir.is_dir():
            continue
        category_embeddings_dir = embeddings_dir / category_dir.name
        category_embeddings_dir.mkdir(parents=True, exist_ok=True)
        for clothing_image in category_dir.iterdir():
            # if clothing_image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            #     continue
            try:
                if clothing_image.is_file():
                    processed_count += 1
                    generator.generate_and_save_embedding(clothing_image, category_embeddings_dir)
            except Exception as e:
                logger.error(f"Failed to process {clothing_image}: {e}")

    logger.info("Finished generating embeddings.")
    logger.info(f"Found {processed_count} images in {image_dir}")

if __name__ == "__main__":
    main()
