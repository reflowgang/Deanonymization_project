"""Top-k cosine/IP search against a precomputed candidate embedding matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    candidate_user_id: str
    rank: int
    score: float


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    try:
        import faiss  # type: ignore

        y = x.astype(np.float32, copy=True)
        faiss.normalize_L2(y)
        return y
    except ModuleNotFoundError:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (x / norms).astype(np.float32)


def search_top_k(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_user_ids: Sequence[str],
    *,
    k: int = 15,
) -> list[list[SearchHit]]:
    if query_vectors.ndim != 2 or candidate_vectors.ndim != 2:
        raise ValueError("Expected 2D embedding matrices")
    if candidate_vectors.shape[0] != len(candidate_user_ids):
        raise ValueError("candidate_user_ids length mismatch")

    q = _normalize_l2(query_vectors)
    c = _normalize_l2(candidate_vectors)
    scores = q @ c.T  # cosine similarity (IP on L2-normalized)
    k_eff = min(k, c.shape[0])

    out: list[list[SearchHit]] = []
    for i in range(scores.shape[0]):
        row = scores[i]
        if k_eff >= row.shape[0]:
            idx = np.argsort(-row)
        else:
            idx = np.argpartition(-row, k_eff - 1)[:k_eff]
            idx = idx[np.argsort(-row[idx])]
        hits: list[SearchHit] = []
        for rank, j in enumerate(idx[:k_eff], start=1):
            hits.append(
                SearchHit(
                    candidate_user_id=str(candidate_user_ids[int(j)]),
                    rank=rank,
                    score=float(row[int(j)]),
                )
            )
        out.append(hits)
    return out
