"""FastAPI composition root for FindMyFit."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from backend.routes import router
from findmyfit.config import Settings
from findmyfit.db.embedding_repository import SqliteEmbeddingRepository
from findmyfit.db.session import create_session_factory
from findmyfit.retrieval.faiss_index import audit_faiss_indexes
from findmyfit.storage.local_images import LocalImageStore

if TYPE_CHECKING:
    from findmyfit import ClothingRecommender


LOGGER = logging.getLogger(__name__)


def _selected_model(settings: Settings) -> tuple[str, str, str]:
    if settings.recommender_engine == "cosine":
        return (
            settings.clip_catalog_model_name,
            settings.clip_model_version,
            "cosine",
        )
    return (
        settings.metric_model_name,
        settings.metric_model_version,
        "l2",
    )


def create_app(
    settings: Settings | None = None,
    recommender_factory: Callable[..., ClothingRecommender] | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    image_store = LocalImageStore(runtime_settings.images_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = runtime_settings
        app.state.image_store = image_store
        app.state.recommender = None
        app.state.component_status = {
            "database": {"ready": False, "detail": "Not initialized"},
            "images": {
                "ready": runtime_settings.images_dir.is_dir(),
                "detail": None
                if runtime_settings.images_dir.is_dir()
                else "Image root is missing",
            },
            "checkpoint": {
                "ready": (
                    runtime_settings.recommender_engine == "cosine"
                    or runtime_settings.metric_checkpoint_path.is_file()
                ),
                "detail": None
                if (
                    runtime_settings.recommender_engine == "cosine"
                    or runtime_settings.metric_checkpoint_path.is_file()
                )
                else "Metric checkpoint is missing",
            },
            "vector_index": {
                "ready": runtime_settings.retrieval_backend == "sqlite",
                "detail": (
                    "SQLite backend selected"
                    if runtime_settings.retrieval_backend == "sqlite"
                    else "Not initialized"
                ),
            },
            "recommender": {"ready": False, "detail": "Not initialized"},
        }
        try:
            selected_factory = recommender_factory
            if selected_factory is None:
                session_factory = create_session_factory(
                    runtime_settings.database_url
                )
                with session_factory() as session:
                    session.execute(select(1))
                app.state.component_status["database"] = {
                    "ready": True,
                    "detail": None,
                }
                if runtime_settings.retrieval_backend == "faiss":
                    model_name, model_version, metric = _selected_model(
                        runtime_settings
                    )
                    index_audit = audit_faiss_indexes(
                        SqliteEmbeddingRepository(session_factory, image_store),
                        root=runtime_settings.faiss_index_dir,
                        model_name=model_name,
                        model_version=model_version,
                        expected_metric=metric,
                        checksums=True,
                    )
                    if not index_audit.ready:
                        app.state.component_status["vector_index"] = {
                            "ready": False,
                            "detail": "FAISS index is missing, stale, or invalid",
                        }
                        raise RuntimeError("FAISS index is not ready")
                    app.state.component_status["vector_index"] = {
                        "ready": True,
                        "detail": None,
                    }
                from findmyfit import ClothingRecommender

                selected_factory = ClothingRecommender
            else:
                app.state.component_status["database"] = {
                    "ready": True,
                    "detail": None,
                }
                app.state.component_status["vector_index"] = {
                    "ready": True,
                    "detail": "Injected recommender factory",
                }
            app.state.recommender = selected_factory(
                recommendation_engine_type=runtime_settings.recommender_engine,
                settings=runtime_settings,
            )
            app.state.component_status["recommender"] = {"ready": True, "detail": None}
        except Exception:
            LOGGER.exception("Recommendation service initialization failed")
            if not app.state.component_status["database"]["ready"]:
                app.state.component_status["database"] = {
                    "ready": False,
                    "detail": "Database or catalog initialization failed",
                }
            app.state.component_status["recommender"] = {
                "ready": False,
                "detail": "Recommendation service initialization failed",
            }
        yield
        app.state.recommender = None

    app = FastAPI(title="FindMyFit API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if runtime_settings.images_dir.is_dir():
        app.mount(
            "/images",
            StaticFiles(directory=runtime_settings.images_dir),
            name="images",
        )
    app.include_router(router)
    return app


app = create_app()
