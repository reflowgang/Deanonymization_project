from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "text-embedding-3-small"
SUMMARY_KEYS_ORDER: list[str] = [
    "interests",
    "topics",
    "writing_style",
    "technical_expertise",
    "personal_attributes",
    "summary",
]


@dataclass(frozen=True)
class Paths:
    extracted_candidate_root: Path
    embeddings_candidate_root: Path
    embeddings_index_csv: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build embeddings for extracted candidate summaries.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of users processed (for testing).",
    )
    return p.parse_args(argv)


def load_api_key_from_env(project_root: Path) -> str:
    env_path = project_root / "api_key.env"
    if not env_path.exists():
        env_path = project_root / ".env"

    load_dotenv(env_path, override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Expected it in {env_path}.")
    return api_key


def get_client(project_root: Path) -> OpenAI:
    _ = load_api_key_from_env(project_root)
    return OpenAI()


def list_candidate_summaries(candidate_dir: Path) -> list[Path]:
    if not candidate_dir.exists():
        return []
    return sorted(p for p in candidate_dir.glob("user_*_summary.json") if p.is_file())


def user_id_from_summary_path(path: Path) -> str:
    # user_XXXX_summary.json -> user_XXXX
    stem = path.stem
    if stem.endswith("_summary"):
        return stem[: -len("_summary")]
    return stem


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        parts: list[str] = []
        for item in v:
            s = flatten_value(item)
            if s:
                parts.append(s)
        return "; ".join(parts)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def summary_to_embedding_text(summary_obj: dict[str, Any]) -> str:
    # Match 04_build_embeddings.py exactly (including extra keys).
    parts: list[str] = []
    for k in SUMMARY_KEYS_ORDER:
        if k in summary_obj:
            val = flatten_value(summary_obj.get(k))
            if val:
                parts.append(f"{k}: {val}")
    for k in sorted(set(summary_obj.keys()) - set(SUMMARY_KEYS_ORDER)):
        val = flatten_value(summary_obj.get(k))
        if val:
            parts.append(f"{k}: {val}")
    return "\n".join(parts).strip()


def embed_text(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=text)
    vec = resp.data[0].embedding
    return np.asarray(vec, dtype=np.float32)


def save_embedding(path: Path, vec: np.ndarray) -> None:
    _ensure_parent_dir(path)
    np.save(path, vec)


def load_embedding_dim(path: Path) -> int:
    vec = np.load(path, mmap_mode="r")
    return int(vec.shape[0])


def main() -> None:
    args = parse_args()
    limit = args.limit

    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        extracted_candidate_root=project_root / "data/extracted_summaries/candidate",
        embeddings_candidate_root=project_root / "data/embeddings/candidate",
        embeddings_index_csv=project_root / "results/tables/hn_candidate_embeddings_index.csv",
    )

    _ensure_dir(paths.embeddings_candidate_root)
    _ensure_parent_dir(paths.embeddings_index_csv)

    print(f"Input summaries: {paths.extracted_candidate_root}")
    print(f"Output vectors:  {paths.embeddings_candidate_root}")
    print(f"Model: {MODEL}")

    summary_files = list_candidate_summaries(paths.extracted_candidate_root)
    if limit is not None:
        summary_files = summary_files[: max(0, limit)]

    print(f"Candidate summaries to consider: {len(summary_files):,}")

    index_rows: list[dict[str, Any]] = []
    client: Optional[OpenAI] = None

    for summary_path in summary_files:
        user_id = user_id_from_summary_path(summary_path)
        out_path = paths.embeddings_candidate_root / f"{user_id}.npy"

        if out_path.exists():
            index_rows.append(
                {
                    "user_id": user_id,
                    "path": str(out_path),
                    "vector_dim": load_embedding_dim(out_path),
                }
            )
            continue

        summary_obj = read_json(summary_path)
        text = summary_to_embedding_text(summary_obj)
        if not text:
            continue

        if client is None:
            client = get_client(project_root)

        vec = embed_text(client, text)
        save_embedding(out_path, vec)

        index_rows.append(
            {
                "user_id": user_id,
                "path": str(out_path),
                "vector_dim": int(vec.shape[0]),
            }
        )

    pd.DataFrame(index_rows).to_csv(paths.embeddings_index_csv, index=False)
    print(f"\nWrote candidate embeddings index: {paths.embeddings_index_csv} ({len(index_rows):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()

