from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class Paths:
    pool_en_raw_dir: Path
    query_profiles_dir: Path
    candidate_profiles_dir: Path
    manifest_csv: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_user_id_from_filename(path: Path, kind: str) -> str:
    # Expected: user_xxxxxxxx_query.jsonl / user_xxxxxxxx_candidate.jsonl
    stem = path.stem  # user_xxxxxxxx_query
    suffix = f"_{kind}"
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected filename pattern for {kind}: {path.name}")
    return stem[: -len(suffix)]


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} in {path}")
            yield obj


def load_sorted_comment_texts(path: Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    for obj in iter_jsonl_records(path):
        if "b" not in obj:
            raise ValueError(f'Missing "b" field in {path}')
        if "t" not in obj:
            raise ValueError(f'Missing "t" field in {path}')

        t_val = obj["t"]
        try:
            t_int = int(t_val)
        except Exception as e:
            raise ValueError(f'Invalid unix timestamp "t"={t_val!r} in {path}') from e

        text = str(obj["b"])
        text = clean_whitespace(text)
        if text:
            rows.append((t_int, text))

    rows.sort(key=lambda x: x[0])
    return [t for _, t in rows]


def write_profile_lines(out_path: Path, lines: list[str]) -> None:
    _ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def count_words_for_lines(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def discover_pool_en_files(raw_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    if not raw_dir.exists():
        return {}, {}

    query_files = sorted(raw_dir.glob("user_*_query.jsonl"))
    cand_files = sorted(raw_dir.glob("user_*_candidate.jsonl"))

    queries: dict[str, Path] = {}
    for p in query_files:
        uid = parse_user_id_from_filename(p, "query")
        queries[uid] = p

    cands: dict[str, Path] = {}
    for p in cand_files:
        uid = parse_user_id_from_filename(p, "candidate")
        cands[uid] = p

    return queries, cands


def infer_role(has_query: bool, has_candidate: bool) -> str:
    if has_query and has_candidate:
        return "matched"
    if has_candidate and not has_query:
        return "distractor"
    if has_query and not has_candidate:
        return "query_only"
    return "unknown"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        pool_en_raw_dir=project_root / "data/raw/POOL-EN",
        query_profiles_dir=project_root / "data/esrc/pool_en/query_profiles",
        candidate_profiles_dir=project_root / "data/esrc/pool_en/candidate_profiles",
        manifest_csv=project_root / "results/tables/pool_en_profile_manifest.csv",
    )

    _ensure_dir(paths.query_profiles_dir)
    _ensure_dir(paths.candidate_profiles_dir)
    _ensure_parent_dir(paths.manifest_csv)

    print(f"Input directory: {paths.pool_en_raw_dir}")
    if not paths.pool_en_raw_dir.exists():
        raise FileNotFoundError(
            f"POOL-EN input folder not found: {paths.pool_en_raw_dir}\n"
            "Place POOL-EN JSONL files there and rerun."
        )

    query_paths, cand_paths = discover_pool_en_files(paths.pool_en_raw_dir)
    all_user_ids = sorted(set(query_paths.keys()) | set(cand_paths.keys()))

    print(f"Found {len(query_paths):,} query JSONL files.")
    print(f"Found {len(cand_paths):,} candidate JSONL files.")
    print(f"Found {len(all_user_ids):,} unique user_ids across both.")

    manifest_rows: list[dict[str, Any]] = []

    for user_id in all_user_ids:
        q_path: Optional[Path] = query_paths.get(user_id)
        c_path: Optional[Path] = cand_paths.get(user_id)

        has_query = q_path is not None
        has_candidate = c_path is not None

        q_lines: list[str] = []
        c_lines: list[str] = []

        if has_query:
            assert q_path is not None
            q_lines = load_sorted_comment_texts(q_path)
            write_profile_lines(paths.query_profiles_dir / f"{user_id}.txt", q_lines)

        if has_candidate:
            assert c_path is not None
            c_lines = load_sorted_comment_texts(c_path)
            write_profile_lines(paths.candidate_profiles_dir / f"{user_id}.txt", c_lines)

        role = infer_role(has_query=has_query, has_candidate=has_candidate)

        manifest_rows.append(
            {
                "user_id": user_id,
                "role": role,
                "has_query": int(has_query),
                "has_candidate": int(has_candidate),
                "n_query_comments": int(len(q_lines)),
                "n_candidate_comments": int(len(c_lines)),
                "query_words": int(count_words_for_lines(q_lines)),
                "candidate_words": int(count_words_for_lines(c_lines)),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["role", "user_id"]).reset_index(drop=True)
    manifest_df.to_csv(paths.manifest_csv, index=False)

    n_query_profiles = int((manifest_df["has_query"] == 1).sum())
    n_candidate_profiles = int((manifest_df["has_candidate"] == 1).sum())
    n_matched = int((manifest_df["role"] == "matched").sum())
    n_distractor = int((manifest_df["role"] == "distractor").sum())
    n_query_only = int((manifest_df["role"] == "query_only").sum())

    print("\n=== POOL-EN profile preparation summary ===")
    print(f"Query profiles written:     {n_query_profiles:,}")
    print(f"Candidate profiles written:{n_candidate_profiles:,}")
    print(f"Matched users:              {n_matched:,}")
    print(f"Distractor users:           {n_distractor:,}")
    if n_query_only:
        print(f"Query-only users:           {n_query_only:,} (unexpected for POOL-EN)")
    print(f"Manifest CSV:               {paths.manifest_csv}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
