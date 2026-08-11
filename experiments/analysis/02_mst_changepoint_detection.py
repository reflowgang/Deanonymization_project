from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt


LEVELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
COMMENT_COUNTS = [5, 10, 25, 50, 100, 200, 500, 500]

REDDIT_ACCURACY = [8.8, 13.4, 16.6, 25.2, 28.2, 33.2, 39.8, 39.4]
HN_ACCURACY = [4.8, 8.6, 15.6, 22.2, 26.0, 32.4, 34.6, 38.0]

MAIN_PENALTY = 2.0
EXPLORATORY_PENALTY = 0.5


def filter_breakpoints(bkps: list[int], n_samples: int) -> list[int]:
    """Drop ruptures endpoint breakpoint when it equals len(signal)."""
    return [int(b) for b in bkps if int(b) < n_samples]


def detect_breakpoints(accuracy: list[float], pen: float) -> list[int]:
    signal = np.array(accuracy, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal)
    bkps = algo.predict(pen=pen)
    return filter_breakpoints(list(bkps), len(accuracy))


def primary_changepoint_index(bkps: list[int], n_samples: int) -> int:
    filtered = filter_breakpoints(bkps, n_samples)
    if not filtered:
        return n_samples - 1
    return filtered[0]


def format_x_labels(levels: list[str], comment_counts: list[int]) -> list[str]:
    return [f"{level}\n{count} comments" for level, count in zip(levels, comment_counts)]


def interpret_changepoint(index: int) -> tuple[str, int]:
    level = LEVELS[index]
    comment_count = COMMENT_COUNTS[index]
    return level, comment_count


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "results/figures"
    tables_dir = project_root / "results/tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_csv = tables_dir / "mst_changepoint_results.csv"
    out_png = figures_dir / "mst_changepoint_detection.png"
    out_pdf = figures_dir / "mst_changepoint_detection.pdf"

    datasets = {
        "Reddit / POOL-EN": REDDIT_ACCURACY,
        "Hacker News": HN_ACCURACY,
    }

    n_samples = len(LEVELS)
    main_results: dict[str, dict[str, object]] = {}
    exploratory_results: dict[str, list[int]] = {}

    for name, accuracy in datasets.items():
        main_bkps_raw = detect_breakpoints(accuracy, pen=MAIN_PENALTY)
        exploratory_bkps = detect_breakpoints(accuracy, pen=EXPLORATORY_PENALTY)
        exploratory_results[name] = exploratory_bkps

        cp_index = primary_changepoint_index(main_bkps_raw, n_samples)
        cp_level, cp_comments = interpret_changepoint(cp_index)

        main_results[name] = {
            "changepoint_index": cp_index,
            "changepoint_level": cp_level,
            "changepoint_comment_count": cp_comments,
            "all_breakpoints_pen2": main_bkps_raw,
        }

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "changepoint_index",
                "changepoint_level",
                "changepoint_comment_count",
            ],
        )
        writer.writeheader()
        for dataset_name, result in main_results.items():
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "changepoint_index": result["changepoint_index"],
                    "changepoint_level": result["changepoint_level"],
                    "changepoint_comment_count": result["changepoint_comment_count"],
                }
            )

    x = np.arange(n_samples)
    x_labels = format_x_labels(LEVELS, COMMENT_COUNTS)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x,
        REDDIT_ACCURACY,
        marker="o",
        linewidth=2,
        label="Reddit / POOL-EN",
    )
    ax.plot(
        x,
        HN_ACCURACY,
        marker="s",
        linewidth=2,
        label="Hacker News",
    )

    reddit_cp = int(main_results["Reddit / POOL-EN"]["changepoint_index"])
    hn_cp = int(main_results["Hacker News"]["changepoint_index"])

    ax.axvline(reddit_cp, color="C0", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axvline(hn_cp, color="C1", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.scatter(
        [reddit_cp],
        [REDDIT_ACCURACY[reddit_cp]],
        color="C0",
        s=120,
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
    )
    ax.scatter(
        [hn_cp],
        [HN_ACCURACY[hn_cp]],
        color="C1",
        s=120,
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
        marker="s",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Minimum Sufficient Text (MST) — Changepoint Detection (PELT, RBF, pen=2)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ymax = max(max(REDDIT_ACCURACY), max(HN_ACCURACY)) * 1.1
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\n=== MST changepoint detection (PELT, model=rbf, pen=2) ===")
    for dataset_name, result in main_results.items():
        print(
            f"{dataset_name}: breakpoints (filtered) = {result['all_breakpoints_pen2']}; "
            f"MST changepoint index = {result['changepoint_index']}"
        )

    print("\n=== Exploratory breakpoints (pen=0.5, filtered) ===")
    for dataset_name, bkps in exploratory_results.items():
        print(f"{dataset_name}: {bkps}")

    reddit_level = main_results["Reddit / POOL-EN"]["changepoint_level"]
    reddit_comments = main_results["Reddit / POOL-EN"]["changepoint_comment_count"]
    hn_level = main_results["Hacker News"]["changepoint_level"]
    hn_comments = main_results["Hacker News"]["changepoint_comment_count"]

    print("\n=== Interpretation ===")
    print(f"Reddit MST changepoint: {reddit_level} / {reddit_comments} comments")
    print(f"Hacker News MST changepoint: {hn_level} / {hn_comments} comments")

    print("\nSaved outputs:")
    print(f"- {out_csv}")
    print(f"- {out_png}")
    print(f"- {out_pdf}")


if __name__ == "__main__":
    main()
