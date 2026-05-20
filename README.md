# Биометрическая идентификация по татуировкам

Backend-сервис на FastAPI для регистрации людей, хранения нескольких фотографий татуировок на одного человека и поиска совпадений через FAISS.

## Что внутри

- `app/main.py` - FastAPI-приложение, API и веб-интерфейс.
- `app/database.py` - SQLite, FAISS-индекс и функции доступа к данным.
- `app/ml_model.py` - загрузка PyTorch-модели и получение 128-мерного embedding-вектора.
- `app/templates/` - HTML-шаблоны интерфейса.
- `app/static/` - CSS интерфейса.
- `NN/resnet.py` - код обучения нейросети.
- `models/tattoo_embedding.pth` - ожидаемый путь к весам модели для запуска приложения.
- `data/` - локальная SQLite-база и FAISS-файлы, создаются автоматически.
- `uploaded_photos/` - локальное хранилище загруженных изображений.

## Модель

Приложение ожидает checkpoint, совместимый с `NN/resnet.py`.

```text
Tattoo Identification Embedding Network
Architecture : ResNet-18 (ImageNet pretrained) -> projector(512->256->128) + L2-norm
Loss          : Triplet Loss (margin = 0.2)
Embedding dim : 128
```

Файл весов должен лежать здесь:

```bash
models/tattoo_embedding.pth
```

Если веса лежат в `NN/tattoo_embedder.pth`, скопируйте их в runtime-папку:

```bash
cp NN/tattoo_embedder.pth models/tattoo_embedding.pth
```

Для дообучения модели на GPU в Google Colab используйте инструкцию:

```text
COLAB_TRAINING.md
```

Для Colab уже подготовлен один архив:

```text
tattoo_colab_package.zip
```

Его нужно загрузить в Google Drive, распаковать по инструкции и запустить обучение. Итоговый файл весов после обучения будет называться правильно:

```text
models/tattoo_embedding.pth
```

Проверка перед запуском:

```bash
test -f models/tattoo_embedding.pth && echo "файл модели найден"
```

## Запуск готового Docker-образа

Образ собирается в GitHub Actions и публикуется в GitHub Container Registry.

Скачать готовый образ:

```bash
docker pull ghcr.io/kernyx/ai_project:latest
```

Подготовить локальные папки:

```bash
mkdir -p data uploaded_photos models
```

Запустить контейнер:

```bash
docker run -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploaded_photos:/app/uploaded_photos" \
  -v "$(pwd)/models:/app/models" \
  ghcr.io/kernyx/ai_project:latest
```

После запуска интерфейс доступен по адресу:

```text
http://127.0.0.1:8000/
```

Проверка состояния:

```bash
curl http://127.0.0.1:8000/health
```

## Локальный запуск без Docker

Рекомендуемый Python: `3.11` или `3.12`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Если нужно указать другой путь к весам:

```bash
TATTOO_MODEL_PATH=/absolute/path/to/model.pth uvicorn app.main:app --reload
```

## Локальная сборка Docker

Используйте этот вариант только если нужно специально собрать образ на своей машине:

```bash
docker build -t tattoo-biometric-backend .
docker run -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploaded_photos:/app/uploaded_photos" \
  -v "$(pwd)/models:/app/models" \
  tattoo-biometric-backend
```

## Сборка образа на GitHub

Workflow: `.github/workflows/docker-image.yml`.

Образ пересобирается только когда меняются файлы, влияющие на Docker-образ:

- `app/**`
- `Dockerfile`
- `requirements.txt`
- `.dockerignore`
- `.github/workflows/docker-image.yml`

Изменения документации, датасета, локальной базы, фотографий и весов модели не запускают пересборку образа.

## Проблемы с правами доступа

Если при добавлении человека появляется ошибка `attempt to write a readonly database`, значит контейнер не может записать в примонтированную папку `data/`.

Проверьте, что контейнер запускается с пользователем хоста:

```bash
docker run -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/uploaded_photos:/app/uploaded_photos" \
  -v "$(pwd)/models:/app/models" \
  ghcr.io/kernyx/ai_project:latest
```

Если файлы в `data/` были созданы от `root`, верните владельца своему пользователю:

```bash
sudo chown -R "$(id -u):$(id -g)" data uploaded_photos
```

## API

- `GET /` - веб-интерфейс оператора.
- `GET /health` - статус приложения и модели.
- `POST /api/add_new_person` - создать человека и добавить первое фото.
- `POST /api/add_new_person_batch` - создать человека и добавить несколько фото.
- `POST /api/add_photo_to_existing` - добавить фото существующему человеку.
- `POST /api/add_photos_to_existing` - добавить несколько фото существующему человеку.
- `POST /api/search` - найти человека по фото татуировки.

Все `POST` эндпоинты принимают `multipart/form-data`.

## Что не коммитить

Эти файлы являются локальными runtime-артефактами и уже добавлены в `.gitignore`:

- `models/*.pth`, `models/*.pt`, `models/*.ckpt`
- `NN/*.pth`, `NN/*.pt`, `NN/*.ckpt`
- `NN/data/`
- `NN/*.ipynb`
- `data/biometric.db`
- `data/faiss.index`
- `data/faiss_meta.json`
- `uploaded_photos/*`
- `.venv/`
