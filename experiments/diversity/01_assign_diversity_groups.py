from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GROUPS, LOW_MAX, MEDIUM_MAX, T4_COMMENT_COUNT, paths


def assign_group(unique_subreddits: int) -> str:
    if unique_subreddits <= LOW_MAX:
        return "low"
    if unique_subreddits <= MEDIUM_MAX:
        return "medium"
    return "high"


def count_unique_subreddits(jsonl_path: Path, n_comments: int) -> int:
    subreddits: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_comments:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            subreddit = obj.get("s")
            if subreddit:
                subreddits.add(str(subreddit))
    return len(subreddits)


def main() -> None:
    p = paths()
    manifest_df = pd.read_csv(p["manifest"])
    query_users = manifest_df.loc[manifest_df["has_query"] == 1, "user_id"].astype(str).tolist()

    rows: list[dict[str, object]] = []
    missing_jsonl = 0

    for user_id in sorted(query_users):
        jsonl_path = p["raw_query_jsonl"] / f"{user_id}_query.jsonl"
        if not jsonl_path.exists():
            missing_jsonl += 1
            continue
        unique_subreddits = count_unique_subreddits(jsonl_path, T4_COMMENT_COUNT)
        rows.append(
            {
                "user_id": user_id,
                "unique_subreddits": unique_subreddits,
                "diversity_group": assign_group(unique_subreddits),
            }
        )

    out_path = p["group_manifest"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "unique_subreddits", "diversity_group"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["diversity_group"] for row in rows)
    print(f"Query users in manifest: {len(query_users):,}")
    print(f"Users with JSONL + diversity label: {len(rows):,}")
    if missing_jsonl:
        print(f"Missing JSONL files: {missing_jsonl:,}")

    print("\nUsers per diversity group:")
    for group in GROUPS:
        print(f"- {group}: {counts.get(group, 0):,}")

    print(f"\nWrote manifest: {out_path}")


if __name__ == "__main__":
    main()
