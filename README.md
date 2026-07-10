# FindMyFit

FindMyFit is an AI-powered fashion recommendation app. Upload a clothing item, choose its category, select the categories you want to match against, and get outfit recommendations ranked by compatibility.

The project includes a React frontend, a FastAPI backend, and a Python recommendation layer that uses CLIP embeddings plus a metric-learning model.

## Demo Video


[Watch the demo](https://youtu.be/BaSdn4e1AtY)

## Features

- Upload clothing images from the browser
- Select the uploaded item's category
- Choose which clothing categories to match with
- Generate ranked outfit recommendations
- Serve recommended item images from the local image dataset
- Support CLIP-based and metric-learning recommendation engines

## Tech Stack

- Frontend: React, Vite, Tailwind CSS
- Backend: FastAPI, Uvicorn
- ML: PyTorch, CLIP, NumPy, scikit-learn, Pillow
- Data: local clothing images, embeddings, and model checkpoints

## Project Structure

```text
.
+-- app.py                         # Optional Streamlit app
+-- backend/
|   +-- main.py                    # FastAPI API server
|   +-- requirements.txt           # Backend runtime dependencies
+-- frontend/
|   +-- package.json               # React/Vite scripts and dependencies
|   +-- src/                       # Frontend app source
+-- scripts/                       # Data and embedding preparation scripts
+-- src/
|   +-- data_manager/              # Embedding loading helpers
|   +-- database/                  # Database models and setup scripts
|   +-- fashion_matcher/           # Recommendation domain logic
|   +-- models/metric_learning/    # Metric-learning model and training code
|   +-- utils/                     # Shared utilities
+-- tests/                         # Test files
```

## Prerequisites

- Python 3.10+
- Node.js 20+
- A local `.env` file
- Local image data and generated embeddings
- Metric-learning checkpoint at `checkpoints/metric_learning/best_model.pt`

The app reads `IMAGES_DIR` from `.env` and defaults to `data/images` when it is not set.

Example `.env`:

```env
IMAGES_DIR=data/images
```

The repository intentionally ignores large local data and model artifacts such as `data/`, `checkpoints/`, database files, and model weights.

## Setup

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install -r backend/requirements.txt
```

Install the additional ML dependencies used by the recommender:

```powershell
pip install torch numpy scikit-learn pillow python-dotenv
pip install git+https://github.com/openai/CLIP.git
```

Install the frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Running the App

Start the FastAPI backend from the project root:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the React frontend:

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

The frontend expects the backend at `http://localhost:8000`.

## Optional Streamlit App

This repository also includes a Streamlit entry point:

```powershell
streamlit run app.py
```

Use this if you want a single Python-based interface instead of the React/FastAPI flow.

## Preparing Data

The recommendation system expects local image assets, stored embeddings, and a trained metric-learning checkpoint. Useful scripts live in `scripts/`:

- `scripts/convert_avif.py`
- `scripts/create_clip_embeddings.py`
- `scripts/create_metric_embeddings.py`
- `scripts/create_training_pairs.py`
- `scripts/testing.py`

Database setup and verification helpers live in `src/database/scripts/`.

## API Endpoints

- `GET /` - Health check
- `GET /categories` - Returns supported clothing categories
- `POST /recommend` - Accepts an uploaded image and recommendation options, then returns ranked recommendations

## Testing

Run tests from the project root:

```powershell
pytest
```

## Notes

- Keep `.env` and local datasets out of git.
- Make sure `IMAGES_DIR` points to the image directory used when generating embeddings.
- If the metric-learning checkpoint is missing, the default backend recommender will fail during startup because it loads `checkpoints/metric_learning/best_model.pt`.
