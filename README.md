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

## Docker run

```bash
docker build -t tattoo-biometric-backend .
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploaded_photos:/app/uploaded_photos" \
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
