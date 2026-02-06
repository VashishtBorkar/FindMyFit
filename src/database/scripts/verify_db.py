# src/database/scripts/verify_db.py

from src.database.database import SessionLocal
from src.database.models import Image, Embedding, Model

def verify_database():
    session = SessionLocal()
    
    print("=" * 50)
    print("Database Verification")
    print("=" * 50)
    
    # Count models
    models = session.query(Model).all()
    print(f"\nModels ({len(models)}):")
    for m in models:
        print(f"  - {m.name} ({m.version}): dim={m.embedding_dim}")
    
    # Count images
    image_count = session.query(Image).count()
    print(f"\nTotal images: {image_count}")
    
    # Count by category
    print("\nImages by category:")
    categories = session.query(Image.category, Image.id).all()
    category_counts = {}
    for cat, _ in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}")
    
    # Count embeddings
    embedding_count = session.query(Embedding).count()
    print(f"\nTotal embeddings: {embedding_count}")
    
    # Embeddings by model
    print("\nEmbeddings by model:")
    for model in models:
        count = session.query(Embedding).filter_by(model_id=model.id).count()
        print(f"  - {model.name}: {count}")
    
    # Sample a few images
    print("\nSample images (first 5):")
    sample_images = session.query(Image).limit(5).all()
    for img in sample_images:
        emb_count = session.query(Embedding).filter_by(image_id=img.id).count()
        print(f"  - {img.id} | {img.category} | {emb_count} embeddings")
    
    session.close()
    print("\n" + "=" * 50)


if __name__ == "__main__":
    verify_database()