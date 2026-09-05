from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
LEVELS_ALL: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
TOP_K = 15

@dataclass(frozen=True)
class Paths:
    embeddings_dir: Path
    results_tables_dir: Path
    candidate_embeddings_npy: Path
    candidate_index_csv: Path
    out_csv: Path

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def load_index_csv(path: Path, required_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Index CSV not found: {path}')
    df = pd.read_csv(path)
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(f'Index CSV missing columns {missing}: {path}')
    return df

def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Embeddings .npy not found: {path}')
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D embeddings matrix in {path}, got shape {arr.shape}')
    return arr.astype(np.float32, copy=False)

def normalize_l2_inplace(x: np.ndarray) -> None:
    import faiss
    if x.size == 0:
        return
    faiss.normalize_L2(x)

def build_index_ip(candidate_vectors: np.ndarray):
    import faiss
    dim = int(candidate_vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(candidate_vectors)
    return index

def user_ids_from_index(df: pd.DataFrame) -> list[str]:
    df2 = df.copy()
    df2['row_id'] = df2['row_id'].astype(int)
    df2 = df2.sort_values('row_id', kind='mergesort').reset_index(drop=True)
    if (df2['row_id'].to_numpy() != np.arange(len(df2))).any():
        raise ValueError('Index CSV row_id is not a contiguous 0..N-1 range.')
    return df2['user_id'].astype(str).tolist()

def main() -> None:
    try:
        import faiss
    except ModuleNotFoundError as e:
        raise RuntimeError('faiss is required but not installed. Install faiss-cpu, e.g.\n  pip install faiss-cpu\nThen rerun this script.') from e
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(embeddings_dir=project_root / 'data/esrc/pool_en/embeddings', results_tables_dir=project_root / 'results/tables', candidate_embeddings_npy=project_root / 'data/esrc/pool_en/embeddings/candidate_embeddings.npy', candidate_index_csv=project_root / 'results/tables/pool_en_candidate_embeddings_index.csv', out_csv=project_root / 'results/tables/pool_en_faiss_top15.csv')
    _ensure_parent_dir(paths.out_csv)
    cand_emb = load_embeddings(paths.candidate_embeddings_npy)
    cand_index_df = load_index_csv(paths.candidate_index_csv, required_cols=['row_id', 'user_id', 'summary_path'])
    candidate_user_ids = user_ids_from_index(cand_index_df)
    if cand_emb.shape[0] != len(candidate_user_ids):
        raise ValueError(f'Candidate embeddings rows do not match candidate index rows: {cand_emb.shape[0]} vs {len(candidate_user_ids)}')
    normalize_l2_inplace(cand_emb)
    index = build_index_ip(cand_emb)
    k = min(TOP_K, cand_emb.shape[0])
    results_rows: list[dict[str, object]] = []
    total_queries = 0
    for level in LEVELS_ALL:
        q_emb_path = paths.embeddings_dir / f'query_embeddings_{level}.npy'
        q_index_path = paths.results_tables_dir / f'pool_en_query_embeddings_index_{level}.csv'
        if not q_emb_path.exists() or not q_index_path.exists():
            print(f'- {level}: missing embeddings/index (skipping)')
            continue
        q_emb = load_embeddings(q_emb_path)
        q_index_df = load_index_csv(q_index_path, required_cols=['row_id', 'user_id', 'T_level', 'summary_path'])
        query_user_ids = user_ids_from_index(q_index_df)
        if q_emb.shape[0] != len(query_user_ids):
            raise ValueError(f'{level}: query embeddings rows do not match index rows: {q_emb.shape[0]} vs {len(query_user_ids)}')
        if q_emb.shape[1] != cand_emb.shape[1]:
            raise ValueError(f'{level}: dim mismatch query={q_emb.shape[1]} candidate={cand_emb.shape[1]}')
        normalize_l2_inplace(q_emb)
        (scores, idxs) = index.search(q_emb, k)
        n_queries = len(query_user_ids)
        total_queries += n_queries
        for (qi, q_uid) in enumerate(query_user_ids):
            for rank0 in range(k):
                cand_i = int(idxs[qi, rank0])
                cand_uid = candidate_user_ids[cand_i]
                score = float(scores[qi, rank0])
                results_rows.append({'T_level': level, 'query_user_id': q_uid, 'candidate_user_id': cand_uid, 'rank': int(rank0 + 1), 'score': score})
        print(f'- {level}: queries processed = {n_queries:,}')
    out_df = pd.DataFrame(results_rows, columns=['T_level', 'query_user_id', 'candidate_user_id', 'rank', 'score'])
    out_df.to_csv(paths.out_csv, index=False)
    print(f'\nTotal queries processed: {total_queries:,}')
    print(f'Total rows written: {len(out_df):,}')
    print(f'Output CSV: {paths.out_csv}')
if __name__ == '__main__':
    main()
