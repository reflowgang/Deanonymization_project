from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TRUNCATION_LEVELS: list[str] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]


@dataclass(frozen=True)
class Paths:
    truncated_root: Path
    summaries_root: Path
    profile_lengths_csv: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def clean_profile_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def read_jsonl_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e

            if "text" not in obj:
                raise ValueError(f'Missing "text" field on line {line_no} in {path}')
            texts.append(str(obj["text"]))
    return texts


def build_profile_from_texts(texts: Iterable[str]) -> str:
    raw = " ".join(t for t in texts if t is not None)
    return clean_profile_text(raw)


def count_words(text: str) -> int:
    return len(text.split())


def level_input_files(truncated_level_dir: Path) -> list[Path]:
    if not truncated_level_dir.exists():
        return []
    return sorted(p for p in truncated_level_dir.glob("user_*_query_*.jsonl") if p.is_file())


def user_id_from_filename(path: Path) -> str:
    # Expected: user_XXXX_query_Tk.jsonl
    return path.name.split("_query_")[0]


def write_text(path: Path, text: str) -> None:
    _ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        truncated_root=project_root / "data/truncated",
        summaries_root=project_root / "data/summaries",
        profile_lengths_csv=project_root / "results/tables/hn_profile_lengths.csv",
    )

    _ensure_parent_dir(paths.profile_lengths_csv)
    for level in TRUNCATION_LEVELS:
        _ensure_dir(paths.summaries_root / level)

    print(f"Reading truncated JSONL from: {paths.truncated_root}")
    print(f"Writing text profiles to:    {paths.summaries_root}")

    summary_rows: list[dict] = []

    for level in TRUNCATION_LEVELS:
        in_dir = paths.truncated_root / level
        out_dir = paths.summaries_root / level

        files = level_input_files(in_dir)
        if not files:
            print(f"- {level}: no JSONL files found in {in_dir}")
            summary_rows.append(
                {
                    "level": level,
                    "n_users": 0,
                    "avg_characters": 0.0,
                    "avg_words": 0.0,
                }
            )
            continue

        n_users = 0
        total_chars = 0
        total_words = 0

        for jsonl_path in files:
            user_id = user_id_from_filename(jsonl_path)
            texts = read_jsonl_texts(jsonl_path)
            profile = build_profile_from_texts(texts)

            out_path = out_dir / f"{user_id}.txt"
            write_text(out_path, profile)

            n_users += 1
            total_chars += len(profile)
            total_words += count_words(profile)

        avg_chars = (total_chars / n_users) if n_users else 0.0
        avg_words = (total_words / n_users) if n_users else 0.0

        summary_rows.append(
            {
                "level": level,
                "n_users": int(n_users),
                "avg_characters": float(avg_chars),
                "avg_words": float(avg_words),
            }
        )

        print(
            f"- {level}: users={n_users:,}, avg_chars={avg_chars:,.1f}, avg_words={avg_words:,.1f}"
        )

    pd.DataFrame(summary_rows).to_csv(paths.profile_lengths_csv, index=False)
    print(f"\nWrote summary CSV: {paths.profile_lengths_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

