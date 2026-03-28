import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.fashion_matcher.clothing_recommender import ClothingRecommender

load_dotenv()

images_dir = Path(os.getenv("IMAGES_DIR", "data/images"))

app = FastAPI(title="FindMyFit API")

# Allow your React frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load recommender once at startup
def load_recommender() -> ClothingRecommender:
    return ClothingRecommender(
        recommendation_engine_type="metric",
        images_dir=images_dir
    )


recommender = load_recommender()


# Serve recommendation images from your local images directory
if images_dir.exists():
    app.mount("/images", StaticFiles(directory=images_dir), name="images")


@app.get("/")
def root():
    return {"message": "FindMyFit backend is running"}


@app.get("/categories")
def get_categories():
    try:
        categories = ClothingRecommender.get_allowed_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
async def recommend(
    image: UploadFile = File(...),
    target_category: str = Form(...),
    match_categories: str = Form(...),
    max_recommendations: int = Form(...)
):
    try:
        parsed_match_categories = json.loads(match_categories)

        if not isinstance(parsed_match_categories, list):
            raise ValueError("match_categories must be a list")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="match_categories must be a valid JSON array"
        )

    suffix = Path(image.filename).suffix if image.filename else ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(image.file, tmp_file)

    try:
        results = recommender.get_recommendations(
            image_path=tmp_path,
            target_category=target_category,
            match_categories=parsed_match_categories,
            max_recommendations=max_recommendations
        )

        serialized_results = []

        for idx, rec in enumerate(results):
            rec_path = rec.recommended_item.image_path

            if not rec_path.exists():
                jpg_path = Path(str(rec_path) + ".jpg")
                if jpg_path.exists():
                    rec_path = jpg_path

            image_url = None

            if rec_path.exists():
                try:
                    relative_path = rec_path.relative_to(images_dir)
                    image_url = f"/images/{relative_path.as_posix()}"
                except ValueError:
                    image_url = None

            serialized_results.append({
                "id": idx,
                "category": rec.recommended_item.category,
                "score": round(rec.confidence_score * 100, 1),
                "image": image_url,
                "image_path": str(rec_path)
            })

        return {"recommendations": serialized_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)