from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd
LEVELS_ALL: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']

@dataclass(frozen=True)
class Paths:
    candidate_embeddings_dir: Path
    query_embeddings_root: Path
    out_csv: Path

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_args(argv: Optional[list[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build FAISS index over candidate embeddings and retrieve top-k for each query.')
    p.add_argument('--levels', nargs='+', default=LEVELS_ALL, help='Levels to process, e.g. --levels T1 T2 (default: all).')
    p.add_argument('--top_k', type=int, default=15, help='Number of candidates to retrieve per query (default: 15).')
    return p.parse_args(argv)

def normalize_levels(levels: Iterable[str]) -> list[str]:
    levels_norm = [lvl.strip() for lvl in levels if lvl and lvl.strip()]
    unknown = sorted(set(levels_norm) - set(LEVELS_ALL))
    if unknown:
        raise ValueError(f'Unknown levels: {unknown}. Allowed: {LEVELS_ALL}')
    out: list[str] = []
    seen: set[str] = set()
    for lvl in levels_norm:
        if lvl not in seen:
            out.append(lvl)
            seen.add(lvl)
    return out

def user_id_from_npy(path: Path) -> str:
    return path.stem

def list_npy_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted((p for p in dir_path.glob('user_*.npy') if p.is_file()))

def load_vectors(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    user_ids: list[str] = []
    vectors: list[np.ndarray] = []
    dim: Optional[int] = None
    for p in paths:
        v = np.load(p)
        if v.ndim != 1:
            raise ValueError(f'Expected 1D vector in {p}, got shape {v.shape}')
        if dim is None:
            dim = int(v.shape[0])
        elif int(v.shape[0]) != dim:
            raise ValueError(f'Vector dim mismatch in {p}: expected {dim}, got {v.shape[0]}')
        user_ids.append(user_id_from_npy(p))
        vectors.append(v.astype(np.float32, copy=False))
    if dim is None:
        return ([], np.zeros((0, 0), dtype=np.float32))
    mat = np.vstack(vectors).astype(np.float32, copy=False)
    return (user_ids, mat)

def l2_normalize_inplace(x: np.ndarray) -> None:
    import faiss
    if x.size == 0:
        return
    faiss.normalize_L2(x)

def build_cosine_index(candidate_vectors: np.ndarray):
    import faiss
    if candidate_vectors.ndim != 2 or candidate_vectors.shape[0] == 0:
        raise ValueError('candidate_vectors must be non-empty 2D array')
    dim = int(candidate_vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(candidate_vectors)
    return index

def recall_at_k(query_user_ids: list[str], retrieved_candidate_user_ids: list[list[str]]) -> float:
    if not query_user_ids:
        return 0.0
    hits = 0
    for (qid, cands) in zip(query_user_ids, retrieved_candidate_user_ids):
        if qid in cands:
            hits += 1
    return hits / len(query_user_ids)

def main() -> None:
    args = parse_args()
    levels = normalize_levels(args.levels)
    top_k = int(args.top_k)
    if top_k <= 0:
        raise ValueError('--top_k must be >= 1')
    try:
        import faiss
    except ModuleNotFoundError as e:
        raise RuntimeError('faiss is required but not installed. Install faiss-cpu, e.g.\n  pip install faiss-cpu\nThen rerun this script.') from e
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(candidate_embeddings_dir=project_root / 'data/embeddings/candidate', query_embeddings_root=project_root / 'data/embeddings', out_csv=project_root / 'results/tables/hn_faiss_top15.csv')
    _ensure_parent_dir(paths.out_csv)
    cand_files = list_npy_files(paths.candidate_embeddings_dir)
    if not cand_files:
        print(f'No candidate embeddings found in {paths.candidate_embeddings_dir}.')
        pd.DataFrame([], columns=['level', 'query_user_id', 'rank', 'candidate_user_id', 'similarity', 'is_true_match']).to_csv(paths.out_csv, index=False)
        print(f'Wrote empty results CSV: {paths.out_csv}')
        return
    (candidate_user_ids, candidate_vectors) = load_vectors(cand_files)
    l2_normalize_inplace(candidate_vectors)
    index = build_cosine_index(candidate_vectors)
    n_candidates = len(candidate_user_ids)
    k = min(top_k, n_candidates)
    print(f'Loaded {n_candidates:,} candidate embeddings. dim={candidate_vectors.shape[1]}')
    print(f'Retrieving top_k={k} (requested {top_k}).')
    results_rows: list[dict] = []
    for level in levels:
        query_dir = paths.query_embeddings_root / level
        query_files = list_npy_files(query_dir)
        if not query_files:
            print(f'- {level}: no query embeddings found (skipping level)')
            continue
        (query_user_ids, query_vectors) = load_vectors(query_files)
        if query_vectors.shape[1] != candidate_vectors.shape[1]:
            raise ValueError(f'{level}: query dim {query_vectors.shape[1]} != candidate dim {candidate_vectors.shape[1]}')
        l2_normalize_inplace(query_vectors)
        (sims, idxs) = index.search(query_vectors, k)
        retrieved_ids_for_recall: list[list[str]] = []
        for (qi, q_user_id) in enumerate(query_user_ids):
            cand_ids: list[str] = []
            for rank in range(k):
                cand_idx = int(idxs[qi, rank])
                sim = float(sims[qi, rank])
                cand_user_id = candidate_user_ids[cand_idx]
                cand_ids.append(cand_user_id)
                results_rows.append({'level': level, 'query_user_id': q_user_id, 'rank': rank + 1, 'candidate_user_id': cand_user_id, 'similarity': sim, 'is_true_match': 1 if q_user_id == cand_user_id else 0})
            retrieved_ids_for_recall.append(cand_ids)
        r_at_k = recall_at_k(query_user_ids, retrieved_ids_for_recall)
        print(f'- {level}: n_queries={len(query_user_ids):,}, recall_at_{k}={r_at_k:.4f}')
    pd.DataFrame(results_rows).to_csv(paths.out_csv, index=False)
    print(f'\nWrote results CSV: {paths.out_csv} ({len(results_rows):,} rows)')
if __name__ == '__main__':
    main()
