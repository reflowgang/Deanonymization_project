from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


@dataclass(frozen=True)
class Paths:
    candidate_summaries_dir: Path
    embeddings_dir: Path
    out_embeddings_npy: Path
    out_index_csv: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_summary_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.glob("*.txt") if p.is_file())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        candidate_summaries_dir=project_root / "data/esrc/pool_en/candidate_summaries",
        embeddings_dir=project_root / "data/esrc/pool_en/embeddings",
        out_embeddings_npy=project_root / "data/esrc/pool_en/embeddings/candidate_embeddings.npy",
        out_index_csv=project_root / "results/tables/pool_en_candidate_embeddings_index.csv",
    )

    _ensure_dir(paths.embeddings_dir)
    _ensure_parent_dir(paths.out_index_csv)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "sentence-transformers is required but not installed. Install it, e.g.\n"
            "  pip install sentence-transformers\n"
            "Then rerun this script."
        ) from e

    try:
        from tqdm import tqdm  # type: ignore
    except ModuleNotFoundError:
        tqdm = None  # type: ignore

    print(f"Input candidate summaries: {paths.candidate_summaries_dir}")
    print(f"Output embeddings:         {paths.out_embeddings_npy}")
    print(f"Index CSV:                {paths.out_index_csv}")
    print(f"Model: {MODEL_NAME}")

    files = list_summary_files(paths.candidate_summaries_dir)
    if not files:
        print("No candidate summary .txt files found. Nothing to embed.")
        return

    texts: list[str] = []
    user_ids: list[str] = []
    summary_paths: list[str] = []

    it = files
    if tqdm is not None:
        it = tqdm(files, desc="load candidate")  # type: ignore[assignment]

    for p in it:  # type: ignore[misc]
        text = read_text(p)
        if not text:
            continue
        texts.append(text)
        user_ids.append(p.stem)
        summary_paths.append(str(p))

    if not texts:
        print("All candidate summaries were empty. Nothing to embed.")
        return

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=(tqdm is not None),
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    np.save(paths.out_embeddings_npy, embeddings)

    index_df = pd.DataFrame(
        {
            "row_id": list(range(len(user_ids))),
            "user_id": user_ids,
            "summary_path": summary_paths,
        }
    )
    index_df.to_csv(paths.out_index_csv, index=False)

    print(f"Embedded {len(user_ids):,} candidate summaries.")
    print("Done.")


if __name__ == "__main__":
    main()

