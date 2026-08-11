from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TRUNCATION_LEVELS: list[tuple[str, Optional[int]]] = [
    ("T1", 5),
    ("T2", 10),
    ("T3", 25),
    ("T4", 50),
    ("T5", 100),
    ("T6", 200),
    ("T7", 500),
    ("T8", None),  # full profile
]


@dataclass(frozen=True)
class Paths:
    query_profiles_dir: Path
    truncated_root: Path
    stats_csv: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def try_tqdm(iterable: Iterable[Path], total: int) -> Iterable[Path]:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total)  # type: ignore[return-value]
    except Exception:
        return iterable


def read_nonempty_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            lines.append(line)
    return lines


def count_words_in_lines(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def write_lines(path: Path, lines: list[str]) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(
        query_profiles_dir=project_root / "data/esrc/pool_en/query_profiles",
        truncated_root=project_root / "data/esrc/pool_en/truncated_queries",
        stats_csv=project_root / "results/tables/pool_en_truncation_stats.csv",
    )

    for level, _n in TRUNCATION_LEVELS:
        _ensure_dir(paths.truncated_root / level)

    _ensure_parent_dir(paths.stats_csv)

    print(f"Input query profiles: {paths.query_profiles_dir}")
    if not paths.query_profiles_dir.exists():
        raise FileNotFoundError(f"Query profiles directory not found: {paths.query_profiles_dir}")

    input_files = sorted(paths.query_profiles_dir.glob("*.txt"))
    total_inputs = len(input_files)
    print(f"Found {total_inputs:,} query profile .txt files.")

    written_counts: dict[str, int] = {level: 0 for level, _ in TRUNCATION_LEVELS}

    stats_rows: list[dict[str, object]] = []

    for in_path in try_tqdm(input_files, total=total_inputs):
        user_id = in_path.stem
        all_lines = read_nonempty_lines(in_path)
        n_all = len(all_lines)

        for level, n_take in TRUNCATION_LEVELS:
            if n_take is None:
                selected = all_lines
            else:
                if n_all < n_take:
                    continue
                selected = all_lines[:n_take]

            out_path = paths.truncated_root / level / f"{user_id}.txt"
            write_lines(out_path, selected)

            written_counts[level] += 1
            stats_rows.append(
                {
                    "user_id": user_id,
                    "T_level": level,
                    "n_comments": int(len(selected)),
                    "n_words": int(count_words_in_lines(selected)),
                }
            )

    with paths.stats_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "T_level", "n_comments", "n_words"])
        w.writeheader()
        for row in stats_rows:
            w.writerow(row)

    print("\n=== POOL-EN truncation summary ===")
    print(f"Total input query profiles: {total_inputs:,}")
    for level, _n in TRUNCATION_LEVELS:
        print(f"- {level}: wrote {written_counts[level]:,} files")
    print(f"Stats CSV: {paths.stats_csv}")
    print("==================================\n")


if __name__ == "__main__":
    main()
