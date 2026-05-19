# Tattoo Biometric Backend

## Project structure

- `app/main.py` - FastAPI app and API endpoints
- `app/database.py` - SQLite initialization, FAISS lifecycle, data access helpers
- `app/ml_model.py` - ResNet-18 embedding model loader and inference
- `app/schemas.py` - response schemas
- `app/templates/` - HTML templates for the user interface
- `app/static/` - CSS assets
- `models/tattoo_embedding.pth` - default PyTorch model checkpoint path
- `data/` - SQLite DB and FAISS files, created automatically
- `uploaded_photos/` - uploaded image storage, created automatically
- `.env.example` - local environment variable template

## Native run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

By default the app loads the model checkpoint from:

```bash
models/tattoo_embedding.pth
```

Expected model:

```text
Tattoo Identification Embedding Network
Architecture : ResNet-18 -> FC(512->128) + L2-norm
Loss          : Triplet Loss (margin = 0.2)
Embedding dim : 128
```

To use another checkpoint path:

```bash
TATTOO_MODEL_PATH=/absolute/path/to/model.pth uvicorn app.main:app --reload
```

If you are already inside the `app/` directory, use:

```bash
uvicorn main:app --reload
```

## Run From GitHub Container Registry

The Docker image is built by GitHub Actions and published to GHCR.

Pull the ready image:

```bash
docker pull ghcr.io/kernyx/ai_project:latest
```

Prepare local runtime directories:

```bash
mkdir -p data uploaded_photos models
```

Put the model checkpoint here:

```bash
models/tattoo_embedding.pth
```

Run the downloaded image:

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploaded_photos:/app/uploaded_photos" \
  -v "$(pwd)/models:/app/models" \
  ghcr.io/kernyx/ai_project:latest
```

Image rebuilds are triggered only when Docker-relevant files change: `app/**`, `Dockerfile`, `requirements.txt`, `.dockerignore`, or the Docker workflow itself. Documentation-only changes do not trigger a rebuild.

## Local Docker Build

Use this only if you intentionally want to build the image on your own machine:

```bash
docker build -t tattoo-biometric-backend .
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploaded_photos:/app/uploaded_photos" \
  -v "$(pwd)/models:/app/models" \
  tattoo-biometric-backend
```

## API endpoints

- `GET /` - web interface for operators
- `POST /api/add_new_person`
  - multipart form fields: `full_name`, `file`
- `POST /api/add_photo_to_existing`
  - multipart form fields: `person_id`, `file`
- `POST /api/search`
  - multipart form field: `file`
- `GET /health`
