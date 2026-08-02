# FindMyFit Refactor Plan

> **Status:** The showcase-foundation portion of this plan has been implemented.
> Runtime code now lives in `src/findmyfit/`, offline training lives in
> `training/metric_learning/`, FastAPI has explicit schemas and readiness checks,
> and local paths are configured and auditable. The descriptions of the old
> architecture below are retained as the historical audit that motivated the
> refactor. FAISS, recommendation-logic changes, and ML evaluation fixes remain
> future work.

## 1. Project Overview

FindMyFit currently appears to be an AI-powered clothing recommendation app with two demo frontends:

- Streamlit demo: `app.py`
- React/Vite frontend: `frontend/`
- FastAPI backend: `backend/main.py`
- Recommendation core: `src/fashion_matcher/`
- Database layer: `src/database/`
- Embedding/training scripts: `scripts/`, `src/models/metric_learning/`, `src/data_manager/`
- Local artifacts: `data/findmyfit.db`, `data/clip_embeddings/`, `data/metric_embeddings/`, `checkpoints/`

The main workflow is:

1. User uploads a clothing image.
2. User selects the uploaded item category and categories to match against.
3. Backend generates a CLIP embedding for the uploaded image.
4. Metric engine projects that embedding into the trained FindMyFit metric space.
5. Existing dataset embeddings are loaded from SQLite into memory.
6. Recommendations are scored by distance against matching categories.
7. Backend returns image URLs, categories, and scores to the frontend.

The current database contains:

- `images`: 126,930 rows
- `embeddings`: 253,856 rows
- `models`: `clip/vit-b32` with 512 dimensions and `findmyfit/v1` with 256 dimensions

The repo did not previously have a `docs/` folder. This document now lives at `docs/FINDMYFIT_REFACTOR_PLAN.md`.

## 2. Current Architecture

Current architecture:

- `frontend/src/App.jsx` owns most frontend state, upload preview state, category loading, recommendation fetching, result grouping, and view mode.
- `frontend/src/api.js` talks directly to `http://localhost:8000`.
- `backend/main.py` exposes `GET /categories` and `POST /recommend`, saves uploads to a temp file, calls `ClothingRecommender`, then serializes recommendation objects.
- `app.py` is a separate Streamlit path using the same `ClothingRecommender`.
- `src/fashion_matcher/clothing_recommender.py` validates categories and selects a recommendation engine.
- `src/fashion_matcher/services/recommendation_engine.py` contains database loading, cosine scoring, metric scoring, and a stub BiLSTM engine in one large module.
- `src/fashion_matcher/services/embedding_generator.py` loads CLIP, generates embeddings, and also writes embeddings to the database.
- `src/database/models.py` defines SQLAlchemy models for images, models, and embeddings.
- `src/database/database.py` and `src/database/session.py` both create SQLAlchemy engines/session factories.
- `scripts/create_clip_embeddings.py`, `scripts/create_metric_embeddings.py`, `scripts/create_training_pairs.py`, and `src/database/scripts/migrate_data.py` are the offline embedding/training/data migration path.

Current data flow:

1. React uploads image through `frontend/src/api.js`.
2. FastAPI receives multipart form data in `backend/main.py`.
3. Upload is copied into a temporary local file.
4. `ClothingRecommender.get_recommendations()` builds a `ClothingItem`.
5. `MetricLearningRecommendationEngine` generates CLIP embedding for the uploaded image.
6. It transforms the CLIP vector using `FashionCompatibilityModel` loaded from `checkpoints/metric_learning/best_model.pt`.
7. It scans preloaded metric embeddings from SQLite, filtered by `category_index`.
8. Results are sorted with `heapq.nlargest`.
9. FastAPI maps local image paths under `IMAGES_DIR` to `/images/...`.
10. React displays cards or an outfit layout.

Unclear or confusing areas:

- `data/images` is referenced repeatedly, but the inspected `data/` folder only shows database and embedding folders; image location may depend on `.env`.
- `.env` exists locally, but there is no `.env.example`.
- `src/fashion_matcher/core/config.py` is empty.
- `tests/test_fashion_matcher.py` imports `fashion_matcher.FashionMatcher`, but no matching class was found in the current inspected files.
- `src/models/bilstm/` exists, while the BiLSTM recommendation engine is only a stub.
- `app.py` and `backend/main.py` duplicate app wiring.

## 3. Current Problems

Main issues:

- File organization mixes production backend, Streamlit demo, model training, migration scripts, and recommendation logic without clear boundaries.
- `recommendation_engine.py` combines retrieval, scoring, database loading, model loading, deduplication, and engine classes.
- `embedding_generator.py` both generates embeddings and writes database rows.
- There are duplicate database setup modules: `src/database/database.py` and `src/database/session.py`.
- `src/database/database.py` prints database configuration at import time.
- `backend/main.py`, `app.py`, scripts, and model code hardcode paths such as `data/images`, `data/clip_embeddings`, and `checkpoints/metric_learning/best_model.pt`.
- `frontend/src/api.js` hardcodes `http://localhost:8000`.
- `backend/main.py` hardcodes CORS to `http://localhost:5173`.
- Category constants are duplicated and inconsistent: `outwear` appears in the DB and UI mapping, while `outerwear` appears in `ClothingCategory`; `earrings` differs from enum `EARRING`; `hairwear` has a typo-like enum name `HARIWEAR`.
- Embeddings are stored as raw `LargeBinary` blobs in SQLite, requiring Python-side deserialization and scanning.
- The active recommendation path loads all embeddings into memory at startup, which is risky at the current DB size.
- There is no vector index, nearest-neighbor layer, or DB-level similarity search.
- Upload validation is minimal: file type, image readability, size limits, category constraints, and recommendation count bounds are weak.
- API errors mostly return raw exception strings.
- The frontend does its own recommendation balancing and category-to-slot mapping instead of receiving a clear recommendation contract.
- Test coverage is stale or broken; `pytest` was not installed in the active shell, and the existing test references missing symbols.
- `npm run lint` currently fails in `frontend/src/components/OutfitView.jsx` because `setSelectedIndexes` is called synchronously inside an effect.
- Documentation is mostly absent; `frontend/README.md` is still a Vite template README.

## 4. Recommendation Workflow Review

Image processing:

- Current: uploaded image is stored as a temp file; PIL/CLIP opens it inside `CLIPEmbeddingGenerator`.
- Risk: no central image validation, no size limits, no upload storage abstraction, and temp-file handling is duplicated between Streamlit and FastAPI.
- Cleaner version: add an `ImageService` that validates, normalizes, optionally stores, and returns an internal image reference.

Embedding generation:

- Current: CLIP model loads inside `CLIPEmbeddingGenerator`; metric engine also loads the trained PyTorch projector.
- Risk: model configuration is hardcoded, model load failures happen at import/startup, and embedding generation is coupled to persistence.
- Cleaner version: separate `ClipEmbedder`, `MetricProjector`, and `EmbeddingPipeline`.

Embedding storage:

- Current: SQLite stores embedding vectors as binary blobs in `embeddings.vector`.
- Risk: this works for persistence but not efficient retrieval or similarity search.
- Cleaner version: keep clothing metadata in relational tables and move searchable vectors into pgvector, Chroma, FAISS, or another vector index.

Similarity/retrieval:

- Current: `DatabaseEmbeddingLoader` loads all embeddings and builds a category index in Python, then every request scans candidate categories.
- Risk: startup cost and memory grow with the dataset; no approximate nearest-neighbor index; difficult to scale.
- Cleaner version: expose a `VectorIndex` interface with `query(vector, filters, top_k)` and support metadata filters like category/model_version.

Ranking:

- Current: metric distance becomes `1 / (1 + dist)` and top items are selected by score.
- Risk: ranking is only pairwise vector distance; no clear calibration, no diversity policy beyond hash deduplication, and no category balancing on the backend.
- Cleaner version: split retrieval from ranking: retrieve top candidates per category, dedupe by hash, optionally rerank for compatibility, diversity, and category quotas.

Results:

- Current: backend returns `id`, `category`, `score`, `image`, and local `image_path`.
- Risk: exposing local paths is leaky and frontend must infer presentation logic.
- Cleaner version: return a stable API response with `item_id`, `category`, `score`, `image_url`, `rank`, `model_version`, and optional explanation/debug fields.

## 5. Improved Target Architecture

Recommended structure:

- `backend/app/`: FastAPI app, routes, schemas, dependency wiring, error handlers.
- `src/fashion_matcher/core/`: shared domain models, category definitions, configuration.
- `src/fashion_matcher/services/`: image validation, embedding pipeline, recommendation service.
- `src/fashion_matcher/retrieval/`: vector index interface and implementations.
- `src/fashion_matcher/recommenders/`: cosine, metric, and future outfit-level recommenders.
- `src/database/`: one SQLAlchemy setup module, models, migrations, repositories.
- `scripts/`: offline commands only, using shared services rather than duplicating logic.
- `frontend/src/`: separate `api/`, `components/`, `features/recommendations/`, and shared config.

API layer:

- `GET /health`
- `GET /categories`
- `POST /recommendations`
- Optional future: `GET /items/{id}`, `POST /admin/reindex`, `GET /models`

Services layer:

- `RecommendationService`: request validation, orchestration, response formatting.
- `EmbeddingPipeline`: CLIP image embedding plus metric projection.
- `CatalogRepository`: clothing metadata and image lookup.
- `VectorSearchRepository`: nearest-neighbor search by embedding and metadata filters.

Database schema:

- Keep SQL tables for clothing metadata: item id, category, image path/storage key, hash, source dataset, created/updated timestamps.
- Keep model metadata: name, version, embedding dimensions, checkpoint path, status.
- Add explicit embedding metadata: item id, model id, vector location/index id, content hash, generated_at.
- Prefer migrations over ad hoc `create_db.py`.

Image storage:

- Local development: `data/images/` or a configured local storage root.
- Production: object storage such as S3/R2/Supabase Storage.
- Store only stable storage keys/URLs in metadata, not environment-specific local paths.

Configuration/env:

- Add a typed settings module.
- Add `.env.example`.
- Configure database URL, image root/storage backend, API CORS origins, model checkpoint path, vector backend, and frontend API base URL.
- Remove import-time prints and scattered `load_dotenv()` calls.

Deployment:

- Backend: FastAPI service with model/index loaded at startup.
- Frontend: Vite static build deployed separately.
- Database/vector layer: start with local development, then move to managed Postgres + pgvector or managed vector DB.
- Offline jobs: separate indexing/training commands, not part of request handling.

## 6. Vector Database / Embedding Strategy

Options:

- SQL/NoSQL only: simplest, but not recommended for this dataset because similarity search happens in Python and will not scale cleanly.
- SQLite plus FAISS: good local/demo option; fast, simple, resume-friendly; metadata still stays in SQLite.
- Chroma: good local vector DB for experimentation; easy persistence and metadata filtering, but less resume-production friendly than Postgres.
- pgvector: best practical long-term fit if the project already needs relational metadata and a clean GitHub/resume architecture.
- Pinecone: strong managed vector search, but adds external service dependency and cost.
- Weaviate: powerful managed/self-hosted vector DB, but likely heavier than needed.
- FAISS only: excellent search library, but requires separate metadata/index persistence discipline.

Recommendation:

- Short term: introduce a `VectorIndex` interface and implement a local FAISS or Chroma backend for fast development.
- Best long-term/resume option: use Postgres with `pgvector` so clothing metadata and embeddings can live in one explainable system with indexed vector search.
- Keep clothing metadata in SQL.
- Store searchable embeddings in pgvector or a local FAISS/Chroma index.
- Store one embedding per `(item_id, model_version)`.
- Use image hash plus model version to avoid recomputing embeddings unnecessarily.
- Regenerate embeddings through an explicit reindex command when the model version, checkpoint, preprocessing, or source image hash changes.

### ML cleanup data-source decision

The current local SQLite catalog stores the actual CLIP and metric vectors as
binary blobs. It does not point to the per-item `.npy` files. Runtime
recommendations load the selected model's catalog vectors from SQLite, while
image rows contain portable keys that are resolved beneath the configured
external `IMAGES_DIR`.

The individual CLIP and metric `.npy` files are historical intermediate
artifacts in the offline pipeline:

- CLIP generation writes one `.npy` file per image.
- Metric training reads the CLIP `.npy` files so it does not rerun CLIP during
  every epoch.
- Metric generation projects those CLIP files through the trained checkpoint
  and writes metric `.npy` files.
- The legacy database importer reads both sets of files and copies their vectors
  into SQLite.

The ML cleanup should remove this per-item file dependency:

- Treat the versioned vectors in SQLite as the canonical local source for
  existing catalog embeddings.
- Load CLIP training features from SQLite in batches, or generate one
  consolidated derived training artifact such as an `.npz` or memory-mapped
  array if profiling shows that SQLite is too slow.
- Keep compatibility labels and item/outfit identifiers separate from vector
  storage so train, validation, and test splits can be created before pair
  materialization.
- Write newly projected metric vectors directly to the configured catalog or
  vector index in batches, validating vector dimensions against checkpoint
  metadata.
- Build FAISS indexes from the canonical catalog vectors rather than requiring
  the per-item `.npy` directories.
- Keep an explicit export command for interoperability or debugging; exported
  `.npy` files are reproducible derivatives, not source-of-truth artifacts.
- Continue using image hashes plus model and preprocessing versions to decide
  when a catalog vector must be regenerated.

Do not delete the existing CLIP files until the SQLite-backed training loader or
export path has been implemented and verified. Metric `.npy` files are not
needed by training and are redundant once their vectors are present in the
catalog.

## 7. Refactor Roadmap

Phase 1: Documentation and safety baseline

- Create `docs/FINDMYFIT_REFACTOR_PLAN.md`.
- Add a real root README later, replacing the Vite-only frontend README.
- Add `.env.example`.
- Document current commands, required local data, and model artifacts.
- Fix or replace stale tests.

Phase 2: Configuration and boundaries

- Introduce typed settings.
- Consolidate database session creation into one module.
- Move category definitions into one shared source.
- Replace hardcoded frontend/backend URLs with environment config.
- Add request/response schemas for FastAPI.

Phase 3: Recommendation service cleanup

- Split `recommendation_engine.py` into loader, vector retrieval, scoring, and engine modules.
- Separate CLIP embedding generation from database persistence.
- Add image validation and typed recommendation errors.
- Stop returning local filesystem paths to the frontend.

Phase 4: Retrieval layer

- Add `VectorIndex` interface.
- Implement a local FAISS or Chroma index first.
- Add indexing script that builds or refreshes the vector index from existing DB embeddings.
- Preserve the current SQLite scan path temporarily as a fallback.

Phase 5: pgvector-ready architecture

- Add Postgres/pgvector schema plan.
- Migrate embeddings from binary blobs to vector columns or a dedicated vector store.
- Add category-filtered nearest-neighbor query.
- Add model/version-aware index refresh.

Phase 6: Frontend contract cleanup

- Move API base URL to env.
- Create typed-ish frontend API adapter shape.
- Move recommendation grouping/category-slot logic behind clearer backend response fields or shared constants.
- Keep UI polish secondary to stable workflow.

Phase 7: ML correctness and artifact cleanup

- Replace the per-item `.npy` training dependency with a SQLite-backed loader or
  a consolidated, reproducible training artifact.
- Split by outfit or item group before generating pairs to prevent leakage
  across train, validation, and test datasets.
- Make positive and negative sampling, symmetric-pair handling, and random seeds
  explicit.
- Add retrieval-oriented evaluation metrics and select decision thresholds from
  validation data.
- Record model architecture, preprocessing, embedding version, and evaluation
  results in checkpoint metadata.
- Retrain and version the model before regenerating metric catalog vectors.
- Add FAISS only after the corrected exact-retrieval baseline is covered by
  parity tests.

## 8. Codex Task Breakdown

Task: Create refactor documentation file

Goal: Add the analyzed plan to `docs/FINDMYFIT_REFACTOR_PLAN.md`.

Prompt: "Create `docs/FINDMYFIT_REFACTOR_PLAN.md` from the approved plan. Do not change code."

Task: Add environment documentation

Goal: Add `.env.example` and document required variables without exposing local secrets.

Prompt: "Inspect env usage and create `.env.example` plus README notes for FindMyFit configuration. Do not change runtime behavior."

Task: Consolidate category definitions

Goal: Replace duplicated category sets with one shared constant/module.

Prompt: "Refactor category validation so backend, recommender, and UI-facing category API use one canonical category source. Preserve current category names from the database."

Task: Consolidate database session setup

Goal: Remove duplicate SQLAlchemy engine/session modules.

Prompt: "Consolidate `src/database/database.py` and `src/database/session.py` into one database session module, update imports, and remove import-time prints."

Task: Add API schemas and validation

Goal: Make FastAPI request/response contracts explicit.

Prompt: "Add Pydantic schemas and validation for recommendation requests/responses, including category validation, max result bounds, upload type checks, and safe error responses."

Task: Split recommendation engine responsibilities

Goal: Separate embedding generation, retrieval, scoring, and orchestration.

Prompt: "Refactor `recommendation_engine.py` into small modules for embedding loading, scoring, metric projection, and recommendation orchestration while preserving current behavior."

Task: Add vector index abstraction

Goal: Prepare for FAISS/Chroma/pgvector without committing to one immediately.

Status: Superseded. No speculative public `VectorIndex` abstraction was added.
The concrete FAISS implementation introduced the small internal `VectorSearch`
protocol only when a second retrieval backend existed.

Task: Add local FAISS or Chroma retrieval

Goal: Replace full Python scan with indexed vector search.

Prompt: "Implement a local vector search backend behind the `VectorIndex` interface using the approved dependency, with category filters and model-version awareness."

Task: Clean frontend API configuration

Goal: Remove hardcoded API base URL and improve API error handling.

Prompt: "Move frontend API base URL to Vite env config, centralize API error handling, and preserve current UI behavior."

Task: Repair tests

Goal: Establish a small but real regression suite.

Prompt: "Replace stale tests with focused tests for category validation, recommendation response serialization, embedding loader behavior, and FastAPI validation. Do not require the full image dataset."

## 10. Open Questions

- Should `outwear` remain the canonical category because it is already in the database, or should the project migrate toward `outerwear`?
- Is the React/Vite frontend now the primary UI, with Streamlit treated as legacy/demo-only?
- Is local-first development the priority, or should the target architecture assume cloud deployment soon?
- Are you comfortable adding Postgres/pgvector eventually, or should the project stay file/local-only for resume/demo simplicity?
- Should recommendations be "visually similar," "compatible/complementary," or both as separate modes?
- Should uploaded user images ever be stored, or should they remain temporary request-only files?
- Which vector backend should be implemented first after the abstraction: FAISS, Chroma, or pgvector?
- Do you want the trained metric model treated as the main recommender, with CLIP cosine as fallback/debug mode?
- Should the frontend receive balanced per-category results from the backend, or keep balancing in React?
- Should the old BiLSTM/outfit-generation path be removed, documented as future work, or rebuilt later?

## 11. SQLite-Native and Exact-FAISS Implementation

The storage and retrieval cleanup is now intentionally local-first:

- SQLite is the canonical store for image metadata, model metadata, CLIP vectors,
  and metric vectors.
- CLIP generation writes directly to SQLite and skips current rows before model
  inference. Changed image content invalidates every embedding for that image.
- Metric generation streams CLIP vectors from SQLite, projects only missing rows
  in batches, and upserts results into SQLite.
- Metric-learning data loading reads CLIP vectors from SQLite. Pair generation,
  splitting, loss, and training behavior remain unchanged in this phase.
- Per-item `.npy` files are legacy import inputs only. They are never required by
  the API, current generation scripts, metric generation, or training loader.
- Exact category-partitioned FAISS indexes are rebuildable derivatives of SQLite.
  CLIP uses inner product on normalized vectors; metric retrieval uses squared L2
  with the existing distance-to-score conversion preserved.
- `RETRIEVAL_BACKEND=sqlite` retains the Python linear scan as an explicit recovery
  and benchmark path. Missing or stale FAISS files never trigger silent fallback.

This resolves the earlier question about whether generation and training require
embedding directories: they no longer do. Images remain in the configured external
`IMAGES_DIR`; only relative image keys are stored in SQLite.

Legacy `.npy` directories must not be deleted until the SQLite catalog audit,
SQLite-backed training load, FAISS build/audit, recommendation parity checks, and
real benchmarks all pass. Removal is a manual disk-cleanup checkpoint, not a
migration side effect.

Still deferred:

- train/validation/test leakage and pair-sampling corrections;
- model retraining and checkpoint-version changes;
- approximate FAISS indexes and recall tradeoffs;
- calibrated scores, diversity, and category balancing;
- Postgres/pgvector or cloud deployment architecture.
