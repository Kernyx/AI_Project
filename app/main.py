from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

if __package__ in (None, ""):
    from database import SIMILARITY_THRESHOLD, storage
    from ml_model import get_embedding, get_model_status, load_embedding_model
    from schemas import (
        PersonCreatedBatchResponse,
        PersonCreatedResponse,
        PhotoCreatedBatchResponse,
        PhotoCreatedResponse,
        SearchCandidateResponse,
        SearchFoundResponse,
        SearchNotFoundResponse,
        SearchTopResponse,
    )
else:
    from .database import SIMILARITY_THRESHOLD, storage
    from .ml_model import get_embedding, get_model_status, load_embedding_model
    from .schemas import (
        PersonCreatedBatchResponse,
        PersonCreatedResponse,
        PhotoCreatedBatchResponse,
        PhotoCreatedResponse,
        SearchCandidateResponse,
        SearchFoundResponse,
        SearchNotFoundResponse,
        SearchTopResponse,
    )


app = FastAPI(title="API биометрической идентификации по татуировкам", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@app.on_event("startup")
async def on_startup() -> None:
    await storage.initialize()
    load_embedding_model()


storage.upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploaded_photos", StaticFiles(directory=str(storage.upload_dir)), name="uploaded_photos")


def _looks_like_image(image_bytes: bytes, content_type: str | None) -> str:
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неподдерживаемый тип файла. Разрешены JPEG, PNG, WEBP",
        )

    if content_type == "image/jpeg" and image_bytes.startswith(b"\xff\xd8\xff"):
        return SUPPORTED_IMAGE_TYPES[content_type]
    if content_type == "image/png" and image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return SUPPORTED_IMAGE_TYPES[content_type]
    if content_type == "image/webp" and image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return SUPPORTED_IMAGE_TYPES[content_type]

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Содержимое файла не соответствует заявленному типу изображения",
    )


async def _read_image(file: UploadFile) -> tuple[bytes, str]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загруженный файл пустой",
        )
    extension = _looks_like_image(image_bytes, file.content_type)
    return image_bytes, extension


def _save_image(image_bytes: bytes, suffix: str, extension: str) -> str:
    file_path = storage.upload_dir / f"{suffix}_{uuid4().hex}{extension}"
    file_path.write_bytes(image_bytes)
    return str(file_path)


def _photo_url(photo_path: str) -> str:
    return f"/uploaded_photos/{Path(photo_path).name}"


def _validate_files(files: list[UploadFile]) -> None:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нужно загрузить хотя бы одно фото",
        )


async def _save_photo_for_person(person_id: int, file: UploadFile) -> PhotoCreatedResponse:
    image_bytes, extension = await _read_image(file)
    embedding = get_embedding(image_bytes)
    faiss_id = storage.add_embedding(embedding)
    photo_path = _save_image(image_bytes, f"person_{person_id}", extension)
    photo_id = await storage.save_photo_record(person_id, faiss_id, photo_path)
    return PhotoCreatedResponse(
        photo_id=photo_id,
        person_id=person_id,
        faiss_id=faiss_id,
        photo_path=photo_path,
    )


async def _create_person_with_photos(full_name: str, files: list[UploadFile]) -> PersonCreatedBatchResponse:
    normalized_name = full_name.strip()
    if not normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ФИО не должно быть пустым",
        )

    _validate_files(files)

    try:
        person_id, created_at = await storage.create_person(normalized_name)
        photos = [await _save_photo_for_person(person_id, file) for file in files]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось создать запись человека: {exc}",
        ) from exc

    return PersonCreatedBatchResponse(
        person_id=person_id,
        full_name=normalized_name,
        photo_ids=[photo.photo_id for photo in photos],
        faiss_ids=[photo.faiss_id for photo in photos],
        created_at=created_at,
    )


async def _create_person_with_photo(full_name: str, file: UploadFile) -> PersonCreatedResponse:
    result = await _create_person_with_photos(full_name, [file])
    return PersonCreatedResponse(
        person_id=result.person_id,
        full_name=result.full_name,
        photo_id=result.photo_ids[0],
        faiss_id=result.faiss_ids[0],
        created_at=result.created_at,
    )


async def _add_photos(person_id: int, files: list[UploadFile]) -> PhotoCreatedBatchResponse:
    if not await storage.person_exists(person_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Человек с id={person_id} не найден",
        )

    _validate_files(files)

    try:
        photos = [await _save_photo_for_person(person_id, file) for file in files]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось добавить фото: {exc}",
        ) from exc

    return PhotoCreatedBatchResponse(person_id=person_id, photos=photos)


async def _add_photo(person_id: int, file: UploadFile) -> PhotoCreatedResponse:
    result = await _add_photos(person_id, [file])
    return result.photos[0]


async def _search_person(file: UploadFile) -> SearchFoundResponse | SearchNotFoundResponse:
    result = await _search_top_persons(file, k=1)
    if not result.results or result.results[0].similarity < SIMILARITY_THRESHOLD:
        return SearchNotFoundResponse(status="not_found")

    candidate = result.results[0]

    return SearchFoundResponse(
        status="found",
        person_id=candidate.person_id,
        full_name=candidate.full_name,
        photo_id=candidate.photo_id,
        similarity=candidate.similarity,
    )


async def _search_top_persons(file: UploadFile, k: int = 3) -> SearchTopResponse:
    image_bytes, _ = await _read_image(file)
    embedding = get_embedding(image_bytes)

    try:
        search_hits = await storage.search_top_k(embedding, k=k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось выполнить поиск по индексу: {exc}",
        ) from exc

    candidates: list[SearchCandidateResponse] = []
    for search_hit in search_hits:
        person = await storage.get_person_by_faiss_id(search_hit.faiss_id)
        if person is None:
            continue

        candidates.append(
            SearchCandidateResponse(
                person_id=int(person["person_id"]),
                full_name=str(person["full_name"]),
                photo_id=int(person["photo_id"]),
                faiss_id=int(person["faiss_id"]),
                photo_path=str(person["photo_path"]),
                similarity=search_hit.distance,
                passed_threshold=search_hit.distance >= SIMILARITY_THRESHOLD,
            )
        )

    return SearchTopResponse(
        status="found" if any(candidate.passed_threshold for candidate in candidates) else "not_found",
        threshold=SIMILARITY_THRESHOLD,
        results=candidates,
    )


async def _render_dashboard(
    request: Request,
    *,
    success_message: str | None = None,
    error_message: str | None = None,
    search_result: dict[str, object] | None = None,
) -> HTMLResponse:
    persons = await storage.list_persons()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "persons": persons,
            "success_message": success_message,
            "error_message": error_message,
            "search_result": search_result,
            "threshold": SIMILARITY_THRESHOLD,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    return await _render_dashboard(request)


@app.post("/ui/add_new_person", response_class=HTMLResponse)
async def add_new_person_ui(
    request: Request,
    full_name: str = Form(...),
    file: list[UploadFile] = File(...),
) -> HTMLResponse:
    try:
        result = await _create_person_with_photos(full_name, file)
    except HTTPException as exc:
        return await _render_dashboard(request, error_message=str(exc.detail))

    return await _render_dashboard(
        request,
        success_message=(
            f"Создана запись #{result.person_id}: {result.full_name}. "
            f"Добавлено фото: {len(result.photo_ids)}"
        ),
    )


@app.post("/ui/add_photo_to_existing", response_class=HTMLResponse)
async def add_photo_to_existing_ui(
    request: Request,
    person_id: int = Form(...),
    file: list[UploadFile] = File(...),
) -> HTMLResponse:
    try:
        result = await _add_photos(person_id, file)
    except HTTPException as exc:
        return await _render_dashboard(request, error_message=str(exc.detail))

    return await _render_dashboard(
        request,
        success_message=f"Добавлено фото: {len(result.photos)} к человеку #{result.person_id}",
    )


@app.post("/ui/search", response_class=HTMLResponse)
async def search_ui(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    try:
        result = await _search_top_persons(file, k=3)
    except HTTPException as exc:
        return await _render_dashboard(request, error_message=str(exc.detail))

    if not result.results:
        return await _render_dashboard(
            request,
            search_result={"status": "empty", "results": [], "threshold": SIMILARITY_THRESHOLD},
            success_message="В базе пока нет фото для сравнения",
        )

    return await _render_dashboard(
        request,
        search_result={
            "status": result.status,
            "threshold": result.threshold,
            "results": [
                {
                    "person_id": candidate.person_id,
                    "full_name": candidate.full_name,
                    "photo_id": candidate.photo_id,
                    "similarity": round(candidate.similarity, 4),
                    "passed_threshold": candidate.passed_threshold,
                    "photo_url": _photo_url(candidate.photo_path),
                }
                for candidate in result.results
            ],
        },
        success_message="Поиск завершен",
    )


@app.post("/api/add_new_person", response_model=PersonCreatedResponse, status_code=status.HTTP_201_CREATED)
async def add_new_person(full_name: str = Form(...), file: UploadFile = File(...)) -> PersonCreatedResponse:
    return await _create_person_with_photo(full_name, file)


@app.post(
    "/api/add_new_person_batch",
    response_model=PersonCreatedBatchResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_new_person_batch(
    full_name: str = Form(...),
    file: list[UploadFile] = File(...),
) -> PersonCreatedBatchResponse:
    return await _create_person_with_photos(full_name, file)


@app.post("/api/add_photo_to_existing", response_model=PhotoCreatedResponse, status_code=status.HTTP_201_CREATED)
async def add_photo_to_existing(person_id: int = Form(...), file: UploadFile = File(...)) -> PhotoCreatedResponse:
    return await _add_photo(person_id, file)


@app.post(
    "/api/add_photos_to_existing",
    response_model=PhotoCreatedBatchResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_photos_to_existing(
    person_id: int = Form(...),
    file: list[UploadFile] = File(...),
) -> PhotoCreatedBatchResponse:
    return await _add_photos(person_id, file)


@app.post("/api/search", response_model=SearchFoundResponse | SearchNotFoundResponse)
async def search(file: UploadFile = File(...)) -> SearchFoundResponse | SearchNotFoundResponse:
    return await _search_person(file)


@app.post("/api/search_top", response_model=SearchTopResponse)
async def search_top(file: UploadFile = File(...), k: int = Form(3)) -> SearchTopResponse:
    return await _search_top_persons(file, k=k)


@app.get("/health")
async def healthcheck() -> dict[str, object]:
    return {"status": "ok", "model": get_model_status()}
