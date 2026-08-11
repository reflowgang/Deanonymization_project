from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "text-embedding-3-small"
LEVELS_ALL: list[str] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
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
    extracted_root: Path
    embeddings_root: Path
    embeddings_index_csv: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build embeddings for extracted HN summaries.")
    p.add_argument(
        "--levels",
        nargs="+",
        default=LEVELS_ALL,
        help="Levels to process, e.g. --levels T1 T2 (default: all).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of users per level (for testing).",
    )
    return p.parse_args(argv)


def normalize_levels(levels: Iterable[str]) -> list[str]:
    levels_norm = [lvl.strip() for lvl in levels if lvl and lvl.strip()]
    unknown = sorted(set(levels_norm) - set(LEVELS_ALL))
    if unknown:
        raise ValueError(f"Unknown levels: {unknown}. Allowed: {LEVELS_ALL}")
    out: list[str] = []
    seen: set[str] = set()
    for lvl in levels_norm:
        if lvl not in seen:
            out.append(lvl)
            seen.add(lvl)
    return out


def load_api_key_from_env(project_root: Path) -> str:
    # Requirement: support custom env file name.
    # First try api_key.env, then fall back to .env.
    env_path = project_root / "api_key.env"
    if not env_path.exists():
        env_path = project_root / ".env"

    load_dotenv(env_path, override=False)
    print(f"Loaded env file: {env_path}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found. Expected it in {env_path}.")
    return api_key


def get_client(project_root: Path) -> OpenAI:
    _ = load_api_key_from_env(project_root)
    return OpenAI()


def list_summary_files(level_dir: Path) -> list[Path]:
    if not level_dir.exists():
        return []
    return sorted(p for p in level_dir.glob("user_*_summary.json") if p.is_file())


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
        parts = []
        for item in v:
            s = flatten_value(item)
            if s:
                parts.append(s)
        return "; ".join(parts)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v).strip()


def summary_to_embedding_text(summary_obj: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in SUMMARY_KEYS_ORDER:
        if k in summary_obj:
            val = flatten_value(summary_obj.get(k))
            if val:
                parts.append(f"{k}: {val}")
    # Include any extra keys deterministically (to avoid losing info if schema changes)
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
    levels = normalize_levels(args.levels)
    limit = args.limit

    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        extracted_root=project_root / "data/extracted_summaries",
        embeddings_root=project_root / "data/embeddings",
        embeddings_index_csv=project_root / "results/tables/hn_embeddings_index.csv",
    )

    _ensure_parent_dir(paths.embeddings_index_csv)
    for lvl in LEVELS_ALL:
        _ensure_dir(paths.embeddings_root / lvl)

    print(f"Input summaries: {paths.extracted_root}")
    print(f"Output vectors:  {paths.embeddings_root}")
    print(f"Model: {MODEL}")

    index_rows: list[dict[str, Any]] = []
    client: Optional[OpenAI] = None

    for level in levels:
        in_dir = paths.extracted_root / level
        out_dir = paths.embeddings_root / level

        summary_files = list_summary_files(in_dir)
        if not summary_files:
            print(f"- {level}: no extracted summaries found (skipping level)")
            continue

        if limit is not None:
            summary_files = summary_files[: max(0, limit)]

        print(f"- {level}: {len(summary_files):,} summaries to consider")

        for summary_path in summary_files:
            user_id = user_id_from_summary_path(summary_path)
            out_path = out_dir / f"{user_id}.npy"

            if out_path.exists():
                index_rows.append(
                    {
                        "user_id": user_id,
                        "level": level,
                        "path": str(out_path),
                        "vector_dim": load_embedding_dim(out_path),
                    }
                )
                continue

            summary_obj = read_json(summary_path)
            text = summary_to_embedding_text(summary_obj)
            if not text:
                # Skip empty text; don't create an embedding.
                continue

            if client is None:
                client = get_client(project_root)

            vec = embed_text(client, text)
            save_embedding(out_path, vec)

            index_rows.append(
                {
                    "user_id": user_id,
                    "level": level,
                    "path": str(out_path),
                    "vector_dim": int(vec.shape[0]),
                }
            )

    pd.DataFrame(index_rows).to_csv(paths.embeddings_index_csv, index=False)
    print(f"\nWrote embeddings index: {paths.embeddings_index_csv} ({len(index_rows):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()

