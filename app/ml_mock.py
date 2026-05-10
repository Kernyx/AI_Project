from __future__ import annotations

import numpy as np


EMBEDDING_DIM = 128


def get_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Mock embedding generator for development.

    The input is accepted to keep the function signature compatible with a real
    ML model integration, but it is not used yet.
    """
    _ = image_bytes
    vector = np.random.default_rng().random(EMBEDDING_DIM, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0
    return vector / norm

