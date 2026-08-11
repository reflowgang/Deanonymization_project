"""Embed text with the BSP-compatible sentence-transformers model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def embed_texts(texts: Sequence[str], *, model_name: str = MODEL_NAME) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("sentence-transformers required: pip install sentence-transformers") from e

    model = SentenceTransformer(model_name)
    vecs = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)
