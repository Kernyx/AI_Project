from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms


EMBEDDING_DIM = 128
DEFAULT_MODEL_PATH = Path("models") / "tattoo_embedding.pth"


class TattooEmbeddingNet(nn.Module):
    """
    Tattoo Identification Embedding Network
    Architecture : ResNet-18 (ImageNet pretrained) -> FC(512->128) + L2-norm
    Loss          : Triplet Loss (margin = 0.2)
    Embedding dim : 128
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, EMBEDDING_DIM)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone(images)
        return F.normalize(embeddings, p=2, dim=1)


class TattooProjectorEmbeddingNet(nn.Module):
    """
    Checkpoint-compatible network:
    ResNet-18 convolutional backbone -> projector(512->256->128) + L2-norm.
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, EMBEDDING_DIM),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        embeddings = self.projector(flattened)
        return F.normalize(embeddings, p=2, dim=1)


class TattooEmbeddingService:
    def __init__(self, model_path: Path | None = None) -> None:
        configured_path = os.getenv("TATTOO_MODEL_PATH")
        self.model_path = Path(configured_path) if configured_path else model_path or DEFAULT_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}. "
                "Set TATTOO_MODEL_PATH or place the .pth file at models/tattoo_embedding.pth"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        model = self._build_model_for_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def is_loaded(self) -> bool:
        return self.model is not None

    def embed(self, image_bytes: bytes) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Tattoo embedding model is not loaded")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embedding = self.model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise RuntimeError("Model returned a zero embedding")
        return embedding / norm

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, nn.Module):
            return checkpoint.state_dict()

        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format")

        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return self._normalize_state_dict_keys(candidate)

        return self._normalize_state_dict_keys(checkpoint)

    @staticmethod
    def _build_model_for_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Module:
        if any(key.startswith("projector.") for key in state_dict):
            return TattooProjectorEmbeddingNet()
        return TattooEmbeddingNet()

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = key.removeprefix("module.")
            normalized[clean_key] = value

        if normalized and not any(key.startswith("backbone.") for key in normalized):
            # Support checkpoints saved from torchvision.resnet18 directly.
            normalized = {f"backbone.{key}": value for key, value in normalized.items()}

        return normalized


embedding_service = TattooEmbeddingService()


def load_embedding_model() -> None:
    embedding_service.load()


def get_embedding(image_bytes: bytes) -> np.ndarray:
    return embedding_service.embed(image_bytes)


def get_model_status() -> dict[str, str | bool]:
    return {
        "loaded": embedding_service.is_loaded(),
        "path": str(embedding_service.model_path),
        "device": str(embedding_service.device),
    }
