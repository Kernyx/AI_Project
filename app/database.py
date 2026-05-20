from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import faiss
import numpy as np

if __package__ in (None, ""):
    from ml_model import EMBEDDING_DIM
else:
    from .ml_model import EMBEDDING_DIM


DB_DIR = Path("data")
DB_PATH = DB_DIR / "biometric.db"
FAISS_PATH = DB_DIR / "faiss.index"
FAISS_META_PATH = DB_DIR / "faiss_meta.json"
UPLOAD_DIR = Path("uploaded_photos")
SIMILARITY_THRESHOLD = 0.75


@dataclass(slots=True)
class SearchHit:
    distance: float
    faiss_id: int


class Storage:
    def __init__(self) -> None:
        self.db_path = DB_PATH
        self.faiss_path = FAISS_PATH
        self.faiss_meta_path = FAISS_META_PATH
        self.upload_dir = UPLOAD_DIR
        self.index: faiss.IndexIDMap | None = None
        self.next_faiss_id = 1

    async def initialize(self) -> None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        await self._init_sqlite()
        self._init_faiss()

    async def _init_sqlite(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    faiss_id INTEGER NOT NULL UNIQUE,
                    photo_path TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
                """
            )
            await db.commit()

    def _init_faiss(self) -> None:
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        else:
            base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.index = faiss.IndexIDMap(base_index)
            faiss.write_index(self.index, str(self.faiss_path))

        if self.faiss_meta_path.exists():
            meta = json.loads(self.faiss_meta_path.read_text(encoding="utf-8"))
            self.next_faiss_id = int(meta.get("next_faiss_id", 1))
        else:
            self._persist_faiss_meta()

    def _persist_faiss(self) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        faiss.write_index(self.index, str(self.faiss_path))
        self._persist_faiss_meta()

    def _persist_faiss_meta(self) -> None:
        self.faiss_meta_path.write_text(
            json.dumps({"next_faiss_id": self.next_faiss_id}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    async def create_person(self, full_name: str) -> tuple[int, str]:
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            cursor = await db.execute(
                "INSERT INTO persons (full_name, created_at) VALUES (?, ?)",
                (full_name, created_at),
            )
            await db.commit()
            person_id = cursor.lastrowid
        if person_id is None:
            raise RuntimeError("Failed to create person record")
        return int(person_id), created_at

    async def person_exists(self, person_id: int) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,))
            row = await cursor.fetchone()
        return row is not None

    async def list_persons(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    persons.id AS person_id,
                    persons.full_name AS full_name,
                    persons.created_at AS created_at,
                    COUNT(photos.id) AS photos_count
                FROM persons
                LEFT JOIN photos ON photos.person_id = persons.id
                GROUP BY persons.id, persons.full_name, persons.created_at
                ORDER BY persons.id DESC
                """
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    def add_embedding(self, embedding: np.ndarray) -> int:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        faiss_id = self.next_faiss_id
        vector = np.asarray([embedding], dtype=np.float32)
        ids = np.asarray([faiss_id], dtype=np.int64)
        self.index.add_with_ids(vector, ids)
        self.next_faiss_id += 1
        self._persist_faiss()
        return faiss_id

    async def save_photo_record(self, person_id: int, faiss_id: int, photo_path: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            cursor = await db.execute(
                """
                INSERT INTO photos (person_id, faiss_id, photo_path)
                VALUES (?, ?, ?)
                """,
                (person_id, faiss_id, photo_path),
            )
            await db.commit()
            photo_id = cursor.lastrowid
        if photo_id is None:
            raise RuntimeError("Failed to create photo record")
        return int(photo_id)

    async def search(self, embedding: np.ndarray) -> SearchHit | None:
        hits = await self.search_top_k(embedding, k=1)
        return hits[0] if hits else None

    async def search_top_k(self, embedding: np.ndarray, k: int = 3) -> list[SearchHit]:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        if self.index.ntotal == 0 or k <= 0:
            return []

        vector = np.asarray([embedding], dtype=np.float32)
        search_k = min(k, self.index.ntotal)
        distances, ids = self.index.search(vector, search_k)

        hits: list[SearchHit] = []
        for distance, faiss_id in zip(distances[0], ids[0], strict=False):
            normalized_faiss_id = int(faiss_id)
            if normalized_faiss_id == -1:
                continue
            hits.append(SearchHit(distance=float(distance), faiss_id=normalized_faiss_id))
        return hits

    async def get_person_by_faiss_id(self, faiss_id: int) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    persons.id AS person_id,
                    persons.full_name AS full_name,
                    photos.id AS photo_id,
                    photos.faiss_id AS faiss_id,
                    photos.photo_path AS photo_path
                FROM photos
                INNER JOIN persons ON persons.id = photos.person_id
                WHERE photos.faiss_id = ?
                """,
                (faiss_id,),
            )
            # JOIN is needed because FAISS stores only vector ids, while the actual
            # business entity lives in persons and the vector reference lives in photos.
            row = await cursor.fetchone()

        return dict(row) if row is not None else None

    async def get_photo_by_id(self, photo_id: int) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id, person_id, faiss_id, photo_path
                FROM photos
                WHERE id = ?
                """,
                (photo_id,),
            )
            row = await cursor.fetchone()
        return dict(row) if row is not None else None


storage = Storage()
