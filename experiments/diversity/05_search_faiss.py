from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GROUPS, TOP_K, paths


def load_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_index_csv(path: Path, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Index CSV missing columns {missing}: {path}")
    return df


def user_ids_from_index(df: pd.DataFrame) -> list[str]:
    df2 = df.copy()
    df2["row_id"] = df2["row_id"].astype(int)
    df2 = df2.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    if (df2["row_id"].to_numpy() != np.arange(len(df2))).any():
        raise ValueError("Index CSV row_id is not contiguous 0..N-1.")
    return df2["user_id"].astype(str).tolist()


def normalize_l2_inplace(x: np.ndarray) -> None:
    import faiss  # type: ignore

    if x.size:
        faiss.normalize_L2(x)


def build_index_ip(candidate_vectors: np.ndarray):
    import faiss  # type: ignore

    dim = int(candidate_vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(candidate_vectors)
    return index


def recall_at_k(query_user_ids: list[str], retrieved: list[list[str]]) -> float:
    if not query_user_ids:
        return 0.0
    hits = sum(1 for qid, cands in zip(query_user_ids, retrieved) if qid in cands)
    return hits / len(query_user_ids)


def main() -> None:
    try:
        import faiss  # noqa: F401  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("faiss is required. Install with: pip install faiss-cpu") from e

    p = paths()
    p["faiss_out"].parent.mkdir(parents=True, exist_ok=True)

    cand_emb = load_embeddings(p["candidate_embeddings"])
    cand_index_df = load_index_csv(
        p["candidate_index"], required_cols=["row_id", "user_id", "summary_path"]
    )
    candidate_user_ids = user_ids_from_index(cand_index_df)

    if cand_emb.shape[0] != len(candidate_user_ids):
        raise ValueError(
            f"Candidate embeddings/index mismatch: {cand_emb.shape[0]} vs {len(candidate_user_ids)}"
        )

    normalize_l2_inplace(cand_emb)
    index = build_index_ip(cand_emb)
    k = min(TOP_K, cand_emb.shape[0])

    results_rows: list[dict[str, object]] = []
    group_recalls: dict[str, float] = {}

    for group in GROUPS:
        group_emb_dir = p["group_embeddings_root"] / group
        q_emb_path = group_emb_dir / "query_embeddings.npy"
        q_index_path = group_emb_dir / "query_embeddings_index.csv"

        if not q_emb_path.exists() or not q_index_path.exists():
            raise FileNotFoundError(f"Missing embeddings for group '{group}' in {group_emb_dir}")

        q_emb = load_embeddings(q_emb_path)
        q_index_df = load_index_csv(
            q_index_path, required_cols=["row_id", "user_id", "diversity_group", "summary_path"]
        )
        query_user_ids = user_ids_from_index(q_index_df)

        if q_emb.shape[0] != len(query_user_ids):
            raise ValueError(
                f"{group}: query embeddings/index mismatch: "
                f"{q_emb.shape[0]} vs {len(query_user_ids)}"
            )
        if q_emb.shape[1] != cand_emb.shape[1]:
            raise ValueError(
                f"{group}: dim mismatch query={q_emb.shape[1]} candidate={cand_emb.shape[1]}"
            )

        normalize_l2_inplace(q_emb)
        scores, idxs = index.search(q_emb, k)

        retrieved_ids_for_recall: list[list[str]] = []

        for qi, query_user_id in enumerate(query_user_ids):
            cand_ids: list[str] = []
            for rank0 in range(k):
                cand_i = int(idxs[qi, rank0])
                cand_uid = candidate_user_ids[cand_i]
                cand_ids.append(cand_uid)
                results_rows.append(
                    {
                        "diversity_group": group,
                        "query_user_id": query_user_id,
                        "candidate_user_id": cand_uid,
                        "rank": int(rank0 + 1),
                        "score": float(scores[qi, rank0]),
                        "is_true_match": int(query_user_id == cand_uid),
                    }
                )
            retrieved_ids_for_recall.append(cand_ids)

        r_at_k = recall_at_k(query_user_ids, retrieved_ids_for_recall)
        group_recalls[group] = r_at_k
        print(f"- {group}: n_queries={len(query_user_ids):,}, recall_at_{k}={r_at_k:.4f}")

    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(p["faiss_out"], index=False)

    print(f"\nTotal rows written: {len(out_df):,}")
    print(f"Output CSV: {p['faiss_out']}")
    print("\nFAISS Recall@15 by group:")
    for group in GROUPS:
        print(f"  {group}: {100.0 * group_recalls[group]:.1f}%")


if __name__ == "__main__":
    main()
