from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PersonCreatedResponse(BaseModel):
    person_id: int
    full_name: str
    photo_id: int
    faiss_id: int
    created_at: datetime


class PhotoCreatedResponse(BaseModel):
    photo_id: int
    person_id: int
    faiss_id: int
    photo_path: str


class SearchFoundResponse(BaseModel):
    status: Literal["found"]
    person_id: int
    full_name: str
    photo_id: int
    similarity: float


class SearchNotFoundResponse(BaseModel):
    status: Literal["not_found"]

