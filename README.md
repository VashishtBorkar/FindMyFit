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
    -> exact category-partitioned FAISS search
    -> ranking and hash deduplication
    -> API response with portable image URLs
```

The two modes have different meanings:

- `cosine` ranks visual similarity directly in CLIP space.
- `metric` projects CLIP vectors into the learned compatibility space and ranks by
  Euclidean distance.

SQLite remains the durable source of vectors and metadata. FAISS indexes are
disposable search artifacts rebuilt from SQLite. Set `RETRIEVAL_BACKEND=sqlite` to
use the preserved linear scan for recovery or benchmarking.

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

The runtime package is separated into domain models, embedding inference, vector
search, recommenders, database access, and local image storage. Importing a module
does not load CLIP, read the catalog, or mutate the filesystem.

SQLite stores the catalog's actual CLIP and metric vectors as binary blobs. The
generation scripts write vectors directly to SQLite, metric generation streams CLIP
vectors from SQLite, and training reads CLIP features from SQLite. Per-item `.npy`
files are supported only by the explicit one-time legacy importer.

## Local artifact layout

Large artifacts are intentionally ignored by Git. Relative paths in `.env` are
resolved from the repository root.

```text
data/
    findmyfit.db
    images/<category>/<item filename>
    indexes/faiss/
        clip/vit-b32/<category>.faiss
        findmyfit/v1/<category>.faiss
    benchmarks/
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
pip install -e ".[api,ml,retrieval,dev]"
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
python -m findmyfit catalog audit
python -m findmyfit catalog migrate
python -m findmyfit paths audit
python -m findmyfit paths migrate
```

Migration commands are dry runs unless `--apply` is passed. Catalog migration adds
artifact fingerprints and lookup indexes and corrects dimensions from vector byte
lengths and configured model artifacts. Path migration normalizes verified image
keys beneath `IMAGES_DIR`. Both create a timestamped SQLite backup before writing.

```powershell
python -m findmyfit catalog migrate --apply
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
- `GET /health/ready` reports catalog, image, checkpoint, vector-index, and
  recommender readiness.
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
python scripts/migrate_legacy_data.py --clip-dir <legacy-clip-dir> --metric-dir <legacy-metric-dir>
python scripts/verify_database.py
python -m training.metric_learning.train
```

`create_clip_embeddings.py` hashes each image and skips current rows before running
CLIP. If image content changes, all vectors for that item are invalidated before the
new CLIP vector is generated. `create_metric_embeddings.py` projects missing CLIP
rows in batches and resumes safely.

Build and validate the exact indexes after SQLite is ready:

```powershell
python -m findmyfit faiss build --engine all
python -m findmyfit faiss audit
```

Use `--replace` for an intentional atomic rebuild. A missing, corrupt, stale, or
dimension-mismatched index makes FAISS readiness false; the application never
silently falls back.

Legacy `.npy` directories are not removed automatically. Delete them manually only
after all of these checkpoints succeed:

1. CLIP and metric rows are verified in SQLite.
2. Training loads CLIP features from SQLite.
3. Both FAISS indexes build and audit successfully.
4. Recommendation parity and benchmarks pass.

## Performance evaluation

Run the exact-search and end-to-end comparisons against the real local catalog:

```powershell
python -m findmyfit benchmark retrieval
python -m findmyfit benchmark api
```

The retrieval benchmark warms both backends, checks identical ordering and scores
within `1e-6`, and reports initialization, mean, p50, p95, throughput, and speedup.
The API benchmark also includes upload validation, decoding, CLIP inference,
projection, and serialization. Machine-readable results are ignored under
`data/benchmarks/`; the real run generates `docs/FAISS_EVALUATION.md`.

## Tests

The regression suite uses temporary images and SQLite databases; it does not load
the real catalog, CLIP weights, or checkpoint.

```powershell
pytest
```

## Deferred improvements

- training split leakage, sampling correctness, and stronger evaluation metrics
- approximate FAISS indexes such as IVF or HNSW
- category-balanced retrieval and diversity
- score calibration and recommendation explanations
- frontend component restructuring
