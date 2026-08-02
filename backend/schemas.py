"""Explicit HTTP response contracts."""

from pydantic import BaseModel, Field


class CategoriesResponse(BaseModel):
    categories: list[str]


class RecommendationResponseItem(BaseModel):
    item_id: str
    category: str
    score: float = Field(ge=0.0, le=1.0)
    image_url: str | None


class RecommendationsResponse(BaseModel):
    recommendations: list[RecommendationResponseItem]


class ComponentStatus(BaseModel):
    ready: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    components: dict[str, ComponentStatus]
