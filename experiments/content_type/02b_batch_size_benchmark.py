from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "classify_comments",
    Path(__file__).resolve().parent / "02_classify_comments.py",
)
c02 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(c02)

from config import CLASSIFY_MODEL, CONTENT_TYPES, RANDOM_SEED, paths
from io_utils import iter_sorted_comments, load_query_user_ids, select_pilot_users

BENCHMARK_USERS = 5
BATCH_SIZES = [10, 25, 50]
BENCHMARK_CSV = "content_type_batch_size_benchmark.csv"


def benchmark_paths(batch_size: int) -> tuple[Path, Path]:
    root = paths()["manifest"].parent
    return (
        root / f"content_type_benchmark_bs{batch_size}_classifications.csv",
        root / f"content_type_benchmark_bs{batch_size}_classify_log.csv",
    )


def run_batch_size(
    batch_size: int,
    user_ids: list[str],
    client,
    prompt_template: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    classifications_csv, log_csv = benchmark_paths(batch_size)
    for path in (classifications_csv, log_csv):
        if path.exists():
            path.unlink()

    c02.ensure_csv(classifications_csv, c02.CLASSIFICATION_FIELDS)
    c02.ensure_csv(log_csv, c02.LOG_FIELDS)

    prompt_file_log = "prompts/content_type_classification.txt"
    started = time.perf_counter()

    for user_id in user_ids:
        jsonl_path = raw_dir / f"{user_id}_query.jsonl"
        comments = iter_sorted_comments(jsonl_path)

        for start in range(0, len(comments), batch_size):
            batch = comments[start : start + batch_size]
            labels, log_rows = c02.classify_batch_with_retry(
                client=client,
                model=CLASSIFY_MODEL,
                prompt_template=prompt_template,
                user_id=user_id,
                batch=batch,
                log_csv=log_csv,
                prompt_file_log=prompt_file_log,
            )

            for log_row in log_rows:
                c02.append_row(log_csv, c02.LOG_FIELDS, log_row)

            for comment in batch:
                idx = int(comment["comment_index"])
                c02.append_row(
                    classifications_csv,
                    c02.CLASSIFICATION_FIELDS,
                    {
                        "user_id": user_id,
                        "comment_index": idx,
                        "timestamp": int(comment["timestamp"]),
                        "source_line": int(comment["source_line"]),
                        "label": labels[idx],
                        "body": str(comment["body"]),
                    },
                )

            time.sleep(c02.SLEEP_SECONDS)

    runtime_minutes = (time.perf_counter() - started) / 60.0
    class_df = pd.read_csv(classifications_csv)
    log_df = pd.read_csv(log_csv)
    return class_df, log_df, runtime_minutes


def label_shares(class_df: pd.DataFrame) -> dict[str, float]:
    counts = class_df["label"].astype(str).str.upper().value_counts()
    total = int(len(class_df))
    return {
        "personal_share": round(100.0 * counts.get("P", 0) / total, 2),
        "opinion_share": round(100.0 * counts.get("O", 0) / total, 2),
        "topical_share": round(100.0 * counts.get("T", 0) / total, 2),
    }


def count_metrics(log_df: pd.DataFrame) -> dict[str, int]:
    return {
        "api_calls": int((log_df["status"] == "ok").sum()),
        "failures": int((log_df["status"] == "error").sum()),
        "retries": int((log_df["attempt"] > 1).sum() + (log_df["mode"] == "single_fallback").sum()),
    }


def agreement_rate(left: pd.DataFrame, right: pd.DataFrame) -> float:
    merged = left.merge(
        right,
        on=["user_id", "comment_index"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        return 0.0
    same = (merged["label_a"].astype(str).str.upper() == merged["label_b"].astype(str).str.upper()).sum()
    return 100.0 * same / len(merged)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    p = paths()
    user_ids = select_pilot_users(load_query_user_ids(), n_users=BENCHMARK_USERS, seed=RANDOM_SEED)

    print(f"Benchmark users ({BENCHMARK_USERS}): {', '.join(user_ids)}")
    print(f"Batch sizes to test: {BATCH_SIZES}")

    client = c02.get_client(project_root)
    prompt_template = c02.load_prompt_template(p["classify_prompt"])

    results: list[dict[str, object]] = []
    class_by_bs: dict[int, pd.DataFrame] = {}

    for batch_size in BATCH_SIZES:
        print(f"\n=== batch_size={batch_size} ===")
        class_df, log_df, runtime_minutes = run_batch_size(
            batch_size=batch_size,
            user_ids=user_ids,
            client=client,
            prompt_template=prompt_template,
            raw_dir=p["raw_query_jsonl"],
        )
        class_by_bs[batch_size] = class_df
        metrics = count_metrics(log_df)
        shares = label_shares(class_df)

        row = {
            "batch_size": batch_size,
            "users": BENCHMARK_USERS,
            "comments": int(len(class_df)),
            "api_calls": metrics["api_calls"],
            "runtime_minutes": round(runtime_minutes, 2),
            "failures": metrics["failures"],
            "retries": metrics["retries"],
            **shares,
        }
        results.append(row)
        print(
            f"runtime={row['runtime_minutes']:.2f} min, api_calls={row['api_calls']}, "
            f"failures={row['failures']}, retries={row['retries']}, "
            f"P/O/T={shares['personal_share']}/{shares['opinion_share']}/{shares['topical_share']}%"
        )

    out_csv = p["manifest"].parent / BENCHMARK_CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "users",
                "comments",
                "api_calls",
                "runtime_minutes",
                "failures",
                "retries",
                "personal_share",
                "opinion_share",
                "topical_share",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Label agreement across batch sizes ===")
    pairs = [(10, 25), (10, 50), (25, 50)]
    agreements: list[float] = []
    for a, b in pairs:
        rate = agreement_rate(class_by_bs[a], class_by_bs[b])
        agreements.append(rate)
        print(f"  batch {a} vs {b}: {rate:.2f}% agreement")

    min_agreement = min(agreements) if agreements else 100.0
    max_share_delta = 0.0
    for key in ("personal_share", "opinion_share", "topical_share"):
        vals = [row[key] for row in results]
        max_share_delta = max(max_share_delta, max(vals) - min(vals))

    print(f"\nMax label-share spread across batch sizes: {max_share_delta:.2f} pp")
    print(f"Min pairwise label agreement: {min_agreement:.2f}%")
    print(f"\nWrote: {out_csv}")

    fastest = min(results, key=lambda r: float(r["runtime_minutes"]))
    safest = min(results, key=lambda r: (int(r["failures"]), int(r["retries"])))

    labels_stable = min_agreement >= 98.0 and max_share_delta <= 2.0
    if labels_stable:
        recommended = int(fastest["batch_size"])
        for row in results:
            if int(row["failures"]) > int(safest["failures"]):
                continue
            if float(row["runtime_minutes"]) <= float(fastest["runtime_minutes"]) * 1.15:
                recommended = int(row["batch_size"])
    else:
        recommended = 10

    rec_row = next(r for r in results if int(r["batch_size"]) == recommended)

    print("\n=== Recommendation ===")
    print(
        f"Fastest: batch_size={fastest['batch_size']} "
        f"({fastest['runtime_minutes']:.2f} min, {fastest['api_calls']} API calls)"
    )
    if labels_stable:
        print(
            f"Labels are stable across batch sizes (agreement >= {min_agreement:.1f}%, "
            f"share spread <= {max_share_delta:.2f} pp)."
        )
        print(
            f"Recommend batch_size={recommended} for full classification: "
            f"best speed with acceptable reliability."
        )
    else:
        print(
            "Label distributions differ materially across batch sizes; "
            "recommend batch_size=10 for maximum labeling consistency."
        )
        print(f"Recommend batch_size={recommended} for full classification.")

    est_full_hours = float(rec_row["runtime_minutes"]) / BENCHMARK_USERS * 500 / 60
    est_full_calls = int(rec_row["api_calls"]) / (BENCHMARK_USERS * 500) * 250_000
    est_full_cost = est_full_calls * 0.0005
    print(
        f"Estimated full-run classification (500 users) at batch_size={recommended}: "
        f"~{est_full_hours:.1f} hours, ~{int(est_full_calls):,} API calls, ~${est_full_cost:.2f}."
    )


if __name__ == "__main__":
    main()
