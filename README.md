# FindMyFit

FindMyFit recommends catalog items that visually resemble or are compatible with an
uploaded clothing image. The primary demo is a React frontend backed by FastAPI. The
recommendation runtime uses CLIP image embeddings and an optional learned
metric-projection model.

[Watch the demo](https://youtu.be/BaSdn4e1AtY)

## How it works

```text
uploaded image
    -> CLIP embedding
    -> optional metric projection
    -> SQLite catalog embedding scan
    -> ranking and hash deduplication
    -> API response with portable image URLs
```

The two modes have different meanings:

- `cosine` ranks visual similarity directly in CLIP space.
- `metric` projects CLIP vectors into the learned compatibility space and ranks by
  Euclidean distance.

FAISS and changes to the recommendation/training logic are intentionally deferred.

## Repository structure

```text
backend/                 FastAPI composition root, routes, and schemas
frontend/                React/Vite demo
src/findmyfit/           Installable request-time recommendation package
training/metric_learning Offline datasets, loss, training, and tuning
scripts/                 Explicit data preparation and database commands
examples/                Secondary Streamlit demo
tests/                   Dataset-independent regression tests
```

The runtime package is separated into domain models, embedding inference, SQLite
retrieval, recommenders, database setup, and local image storage. Importing a module
does not load CLIP, read the catalog, or mutate the filesystem.

SQLite stores the catalog's actual CLIP and metric vectors as binary blobs. The
per-item `.npy` files shown below are optional offline intermediates used by the
current generation and training scripts; request-time recommendations do not read
them. The planned ML cleanup will load CLIP training features from SQLite, or from
one consolidated derived artifact, so the project no longer depends on hundreds of
thousands of small embedding files.

## Local artifact layout

Large artifacts are intentionally ignored by Git. Relative paths in `.env` are
resolved from the repository root.

```text
data/
    findmyfit.db
    images/<category>/<item filename>
    embeddings/
        clip/<category>/<item id>.npy
        metric/<category>/<item id>.npy
checkpoints/
    metric_learning/best_model.pt
```

Copy [.env.example](.env.example) to `.env` and override paths when the dataset is
stored elsewhere. SQLite image rows should contain keys relative to `IMAGES_DIR`,
such as `shoes/12345.jpg`.

## Installation

Python 3.10 or newer and Node.js 20 or newer are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[api,ml,dev]"
cd frontend
npm install
cd ..
```

Install offline training or Streamlit dependencies only when needed:

```powershell
pip install -e ".[training]"
pip install -e ".[streamlit]"
```

## Check the local environment

Diagnostics never rewrite the catalog:

```powershell
python -m findmyfit doctor
python -m findmyfit paths audit
python -m findmyfit paths migrate
```

The migration command is a dry run unless `--apply` is passed. Applying database
path normalization creates a timestamped backup first, rewrites only verified files
beneath `IMAGES_DIR`, and refuses artifact conflicts.

```powershell
python -m findmyfit paths migrate --apply
```

Review the audit before applying it. The application never migrates data at startup.

## Run the primary demo

Start FastAPI from the repository root:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm run dev
```

Health endpoints:

- `GET /health/live` verifies that the API process is alive.
- `GET /health/ready` reports catalog, image, checkpoint, and recommender readiness.
- `GET /categories` lists canonical database categories.
- `POST /recommend` accepts a multipart image and repeated `match_categories` fields.

If an artifact cannot initialize, the API remains available for diagnostics and
recommendation requests return `503`.

## Streamlit example

```powershell
streamlit run examples/streamlit_app.py
```

This example uses the same `ClothingRecommender` facade and settings as FastAPI.

## Offline commands

```powershell
python scripts/create_database.py
python scripts/create_clip_embeddings.py
python scripts/create_metric_embeddings.py
python scripts/migrate_legacy_data.py
python scripts/verify_database.py
python -m training.metric_learning.train
```

## Tests

The regression suite uses temporary images and SQLite databases; it does not load
the real catalog, CLIP weights, or checkpoint.

```powershell
pytest
```

## Deferred improvements

- FAISS-backed vector retrieval
- SQLite-backed or consolidated training features instead of per-item `.npy` files
- training split leakage, sampling correctness, and stronger evaluation metrics
- category-balanced retrieval and diversity
- score calibration and recommendation explanations
- frontend component restructuring
