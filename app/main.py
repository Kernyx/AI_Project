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
        PersonCreatedResponse,
        PhotoCreatedResponse,
        SearchFoundResponse,
        SearchNotFoundResponse,
    )
else:
    from .database import SIMILARITY_THRESHOLD, storage
    from .ml_model import get_embedding, get_model_status, load_embedding_model
    from .schemas import (
        PersonCreatedResponse,
        PhotoCreatedResponse,
        SearchFoundResponse,
        SearchNotFoundResponse,
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


async def _create_person_with_photo(full_name: str, file: UploadFile) -> PersonCreatedResponse:
    normalized_name = full_name.strip()
    if not normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ФИО не должно быть пустым",
        )

    image_bytes, extension = await _read_image(file)
    embedding = get_embedding(image_bytes)

    try:
        person_id, created_at = await storage.create_person(normalized_name)
        faiss_id = storage.add_embedding(embedding)
        photo_path = _save_image(image_bytes, f"person_{person_id}", extension)
        photo_id = await storage.save_photo_record(person_id, faiss_id, photo_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось создать запись человека: {exc}",
        ) from exc

    return PersonCreatedResponse(
        person_id=person_id,
        full_name=normalized_name,
        photo_id=photo_id,
        faiss_id=faiss_id,
        created_at=created_at,
    )


async def _add_photo(person_id: int, file: UploadFile) -> PhotoCreatedResponse:
    if not await storage.person_exists(person_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Человек с id={person_id} не найден",
        )

    image_bytes, extension = await _read_image(file)
    embedding = get_embedding(image_bytes)

    try:
        faiss_id = storage.add_embedding(embedding)
        photo_path = _save_image(image_bytes, f"person_{person_id}", extension)
        photo_id = await storage.save_photo_record(person_id, faiss_id, photo_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось добавить фото: {exc}",
        ) from exc

    return PhotoCreatedResponse(
        photo_id=photo_id,
        person_id=person_id,
        faiss_id=faiss_id,
        photo_path=photo_path,
    )


async def _search_person(file: UploadFile) -> SearchFoundResponse | SearchNotFoundResponse:
    image_bytes, _ = await _read_image(file)
    embedding = get_embedding(image_bytes)

    try:
        search_hit = await storage.search(embedding)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось выполнить поиск по индексу: {exc}",
        ) from exc

    if search_hit is None or search_hit.distance < SIMILARITY_THRESHOLD:
        return SearchNotFoundResponse(status="not_found")

    person = await storage.get_person_by_faiss_id(search_hit.faiss_id)
    if person is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"В базе данных нет записи для faiss_id={search_hit.faiss_id}",
        )

    return SearchFoundResponse(
        status="found",
        person_id=int(person["person_id"]),
        full_name=str(person["full_name"]),
        photo_id=int(person["photo_id"]),
        similarity=search_hit.distance,
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
    file: UploadFile = File(...),
) -> HTMLResponse:
    try:
        result = await _create_person_with_photo(full_name, file)
    except HTTPException as exc:
        return await _render_dashboard(request, error_message=str(exc.detail))

    return await _render_dashboard(
        request,
        success_message=f"Создана запись #{result.person_id}: {result.full_name}",
    )


@app.post("/ui/add_photo_to_existing", response_class=HTMLResponse)
async def add_photo_to_existing_ui(
    request: Request,
    person_id: int = Form(...),
    file: UploadFile = File(...),
) -> HTMLResponse:
    try:
        result = await _add_photo(person_id, file)
    except HTTPException as exc:
        return await _render_dashboard(request, error_message=str(exc.detail))

    return await _render_dashboard(
        request,
        success_message=f"Фото #{result.photo_id} добавлено к человеку #{result.person_id}",
    )


@app.post("/ui/search", response_class=HTMLResponse)
async def search_ui(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    try:
        result = await _search_person(file)
    except HTTPException as exc:
        return await _render_dashboard(request, error_message=str(exc.detail))

    if isinstance(result, SearchNotFoundResponse):
        return await _render_dashboard(
            request,
            search_result={"status": "not_found"},
            success_message="Совпадение с достаточной уверенностью не найдено",
        )

    photo = await storage.get_photo_by_id(result.photo_id)
    photo_url = _photo_url(str(photo["photo_path"])) if photo is not None else None

    return await _render_dashboard(
        request,
        search_result={
            "status": "found",
            "person_id": result.person_id,
            "full_name": result.full_name,
            "photo_id": result.photo_id,
            "similarity": round(result.similarity, 4),
            "photo_url": photo_url,
        },
        success_message="Поиск завершен",
    )


@app.post("/api/add_new_person", response_model=PersonCreatedResponse, status_code=status.HTTP_201_CREATED)
async def add_new_person(full_name: str = Form(...), file: UploadFile = File(...)) -> PersonCreatedResponse:
    return await _create_person_with_photo(full_name, file)


@app.post("/api/add_photo_to_existing", response_model=PhotoCreatedResponse, status_code=status.HTTP_201_CREATED)
async def add_photo_to_existing(person_id: int = Form(...), file: UploadFile = File(...)) -> PhotoCreatedResponse:
    return await _add_photo(person_id, file)


@app.post("/api/search", response_model=SearchFoundResponse | SearchNotFoundResponse)
async def search(file: UploadFile = File(...)) -> SearchFoundResponse | SearchNotFoundResponse:
    return await _search_person(file)


@app.get("/health")
async def healthcheck() -> dict[str, object]:
    return {"status": "ok", "model": get_model_status()}
