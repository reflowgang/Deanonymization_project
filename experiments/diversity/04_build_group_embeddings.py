from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBED_MODEL, GROUPS, paths


def list_summary_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.glob("*.txt") if p.is_file())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    p = paths()
    p["group_embeddings_root"].mkdir(parents=True, exist_ok=True)
    p["group_embeddings_root"].parent.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e

    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        tqdm = None  # type: ignore

    print(f"Model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    for group in GROUPS:
        summary_dir = p["group_summaries_root"] / group
        files = list_summary_files(summary_dir)
        if not files:
            raise FileNotFoundError(f"No summaries found for group '{group}' in {summary_dir}")

        texts: list[str] = []
        user_ids: list[str] = []
        summary_paths: list[str] = []

        it = files
        if tqdm is not None:
            it = tqdm(files, desc=f"load {group}")  # type: ignore[assignment]

        for file_path in it:  # type: ignore[misc]
            text = read_text(file_path)
            if not text:
                continue
            texts.append(text)
            user_ids.append(file_path.stem)
            summary_paths.append(str(file_path))

        if not texts:
            raise ValueError(f"Group '{group}' has no non-empty summaries.")

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=(tqdm is not None),
            normalize_embeddings=False,
        ).astype(np.float32, copy=False)

        group_emb_dir = p["group_embeddings_root"] / group
        group_emb_dir.mkdir(parents=True, exist_ok=True)

        out_npy = group_emb_dir / "query_embeddings.npy"
        np.save(out_npy, embeddings)

        index_df = pd.DataFrame(
            {
                "row_id": list(range(len(user_ids))),
                "user_id": user_ids,
                "diversity_group": [group] * len(user_ids),
                "summary_path": summary_paths,
            }
        )
        out_index_csv = group_emb_dir / "query_embeddings_index.csv"
        index_df.to_csv(out_index_csv, index=False)

        print(f"- {group}: embedded {len(user_ids):,} summaries -> {out_npy}")

    print("\nDone.")


if __name__ == "__main__":
    main()
