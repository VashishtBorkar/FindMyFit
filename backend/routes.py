"""FindMyFit HTTP routes."""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from backend.schemas import (
    CategoriesResponse,
    ComponentStatus,
    ReadinessResponse,
    RecommendationResponseItem,
    RecommendationsResponse,
)
from findmyfit.core.categories import get_allowed_categories
from findmyfit.errors import (
    FindMyFitError,
    InvalidCategoryError,
    InvalidRecommendationRequest,
)


router = APIRouter()
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUFFIX_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "FindMyFit backend is running"}


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    components = {
        name: ComponentStatus(**status)
        for name, status in request.app.state.component_status.items()
    }
    response = ReadinessResponse(
        ready=(
            request.app.state.recommender is not None
            and all(component.ready for component in components.values())
        ),
        components=components,
    )
    if response.ready:
        return response
    return JSONResponse(status_code=503, content=response.model_dump())


@router.get("/categories", response_model=CategoriesResponse)
def categories() -> CategoriesResponse:
    return CategoriesResponse(categories=get_allowed_categories())


async def _read_valid_image(upload: UploadFile, max_bytes: int) -> tuple[bytes, str]:
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Image must be JPEG, PNG, or WebP",
        )

    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        with Image.open(BytesIO(content)) as decoded:
            decoded.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    return content, SUFFIX_BY_TYPE[content_type]


@router.post("/recommend", response_model=RecommendationsResponse)
async def recommend(
    request: Request,
    image: UploadFile = File(...),
    target_category: str = Form(...),
    match_categories: list[str] = Form(...),
    max_recommendations: int = Form(..., ge=1),
) -> RecommendationsResponse:
    recommender = request.app.state.recommender
    components_ready = all(
        status["ready"] for status in request.app.state.component_status.values()
    )
    if recommender is None or not components_ready:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not ready",
        )

    settings = request.app.state.settings
    if max_recommendations > settings.max_recommendations:
        raise HTTPException(
            status_code=422,
            detail=(
                "max_recommendations must be between 1 and "
                f"{settings.max_recommendations}"
            ),
        )
    content, suffix = await _read_valid_image(image, settings.max_upload_bytes)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(content)
            temporary_path = temporary.name

        results = recommender.get_recommendations(
            image_path=temporary_path,
            target_category=target_category,
            match_categories=match_categories,
            max_recommendations=max_recommendations,
        )

        items = []
        for result in results:
            item = result.recommended_item
            image_url = None
            if item.image_key is not None:
                try:
                    request.app.state.image_store.resolve(item.image_key)
                    image_url = request.app.state.image_store.url_for(item.image_key)
                except (FileNotFoundError, FindMyFitError):
                    image_url = None
            items.append(
                RecommendationResponseItem(
                    item_id=item.id,
                    category=item.category,
                    score=result.confidence_score,
                    image_url=image_url,
                )
            )
        return RecommendationsResponse(recommendations=items)
    except (InvalidCategoryError, InvalidRecommendationRequest) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Required image was not found") from exc
    except FindMyFitError as exc:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is unavailable",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Recommendation failed") from exc
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
