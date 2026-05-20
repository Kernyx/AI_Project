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


class PersonCreatedBatchResponse(BaseModel):
    person_id: int
    full_name: str
    photo_ids: list[int]
    faiss_ids: list[int]
    created_at: datetime


class PhotoCreatedResponse(BaseModel):
    photo_id: int
    person_id: int
    faiss_id: int
    photo_path: str


class PhotoCreatedBatchResponse(BaseModel):
    person_id: int
    photos: list[PhotoCreatedResponse]


class PhotoInfoResponse(BaseModel):
    photo_id: int
    person_id: int
    faiss_id: int
    photo_path: str
    photo_url: str


class DeletePhotoResponse(BaseModel):
    status: Literal["deleted"]
    photo_id: int
    person_id: int
    faiss_id: int


class DeletePersonResponse(BaseModel):
    status: Literal["deleted"]
    person_id: int
    deleted_photos_count: int


class SearchFoundResponse(BaseModel):
    status: Literal["found"]
    person_id: int
    full_name: str
    photo_id: int
    similarity: float


class SearchNotFoundResponse(BaseModel):
    status: Literal["not_found"]


class SearchCandidateResponse(BaseModel):
    person_id: int
    full_name: str
    photo_id: int
    faiss_id: int
    photo_path: str
    similarity: float
    passed_threshold: bool


class SearchTopResponse(BaseModel):
    status: Literal["found", "not_found"]
    threshold: float
    results: list[SearchCandidateResponse]
