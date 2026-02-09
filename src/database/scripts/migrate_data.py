import os
import argparse
from dotenv import load_dotenv
import numpy as np
import hashlib
from pathlib import Path

from src.database.database import SessionLocal
from src.database.models import Image, Embedding, Model

load_dotenv()
images_dir = Path(os.getenv("IMAGES_DIR", Path("data/images")))
clip_emb_dir = Path(os.getenv("CLIP_EMBEDDINGS_DIR", Path("data/clip_embeddings")))
metric_emb_dir = Path(os.getenv("METRIC_EMBEDDINGS_DIR", Path("data/metric_embeddings")))
BATCH_SIZE = 1000  # how often to commit to db


def compute_image_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def add_model(session, name, version, embedding_dim):
    """Insert model row if missing. Return model_id."""
    model = (
        session.query(Model)
        .filter_by(name=name, version=version)
        .one_or_none()
    )

    if model:
        return model.id

    model = Model(
        name=name,
        version=version,
        embedding_dim=embedding_dim
    )
    session.add(model)
    session.commit()
    return model.id


def load_embedding_file(path: Path):
    """Load numpy embedding and convert to raw bytes."""
    arr = np.load(path)
    return arr.astype(np.float32).tobytes()


def migrate_to_db(limit: int = None):
    """
    Migrate images and embeddings to the database.
    
    Args:
        limit: Maximum number of images to migrate. None = migrate all.
    """
    session = SessionLocal()

    clip_model_id = add_model(session, "clip", "vit-b32", 512)
    metric_model_id = add_model(session, "findmyfit", "v1", 256)

    print("CLIP model id:", clip_model_id)
    print("FM model id:", metric_model_id)

    # Debug: Print the directories being used
    print(f"\n--- Directory Paths ---")
    print(f"Images dir: {images_dir} (exists: {images_dir.exists()})")
    print(f"CLIP emb dir: {clip_emb_dir} (exists: {clip_emb_dir.exists()})")
    print(f"Metric emb dir: {metric_emb_dir} (exists: {metric_emb_dir.exists()})")

    print(f"\n--- Sample files in CLIP dir ---")
    if clip_emb_dir.exists():
        for i, f in enumerate(clip_emb_dir.iterdir()):
            if i >= 5:
                break
            print(f"  {f.name}")
    
    if limit:
        print(f"\n*** TEST MODE: Migrating only {limit} images ***\n")

    total_count = 0
    total_clip = 0
    total_metric = 0

    for category_dir in images_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        # Check if we've hit the limit
        if limit and total_count >= limit:
            break

        category = category_dir.name
        print(f"\nProcessing category: {category}")

        category_count = 0

        for img_path in category_dir.iterdir():
            # Check if we've hit the limit
            if limit and total_count >= limit:
                print(f"[INFO] Reached limit of {limit} images. Stopping.")
                break
                
            if not img_path.is_file():
                continue

            image_id = img_path.stem
            img_hash = compute_image_hash(img_path)

            img = Image(
                id=image_id,
                file_path=str(img_path),
                category=category,
                hash=img_hash
            )
            session.merge(img)
            session.flush()

            # CLIP embedding
            clip_emb_path = clip_emb_dir / category / f"{image_id}.npy"
            existing_clip = session.query(Embedding).filter_by(
                image_id=image_id,
                model_id=clip_model_id
            ).one_or_none()

            if not existing_clip and clip_emb_path.exists():
                try:
                    clip_bytes = load_embedding_file(clip_emb_path)
                    session.add(
                        Embedding(
                            image_id=image_id,
                            model_id=clip_model_id,
                            vector=clip_bytes,
                            dim=512,
                            dtype="float32"
                        )
                    )
                    total_clip += 1
                except Exception as e:
                    print(f"[WARN] CLIP failed for {image_id}: {e}")

            # Metric embedding
            metric_emb_path = metric_emb_dir / category / f"{image_id}.npy"
            existing_metric = session.query(Embedding).filter_by(
                image_id=image_id,
                model_id=metric_model_id
            ).one_or_none()

            if not existing_metric and metric_emb_path.exists():
                try:
                    metric_bytes = load_embedding_file(metric_emb_path)
                    session.add(
                        Embedding(
                            image_id=image_id,
                            model_id=metric_model_id,
                            vector=metric_bytes,
                            dim=256,
                            dtype="float32"
                        )
                    )
                    total_metric += 1
                except Exception as e:
                    print(f"[WARN] Metric failed for {image_id}: {e}")

            category_count += 1
            total_count += 1
            
            if total_count % BATCH_SIZE == 0:
                session.commit()
                print(f"[INFO] Committed {total_count} images so far...")

        try: 
            session.commit()
            print(f"[INFO] Completed '{category}': {category_count} images")
        except Exception as e:
            session.rollback()
            print(f"[ERROR] Failed to commit category '{category}': {e}")

    print("\n" + "=" * 50)
    print("Migration Summary:")
    print("=" * 50)
    print(f"  Total images migrated: {total_count}")
    print(f"  CLIP embeddings added: {total_clip}")
    print(f"  Metric embeddings added: {total_metric}")
    print("=" * 50)
    
    session.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate image data to database")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of images to migrate (for testing)"
    )
    args = parser.parse_args()
    
    migrate_to_db(limit=args.limit)
