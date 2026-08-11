from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
COMMENT_COUNTS: list[int | str] = [5, 10, 25, 50, 100, 200, 500, "Full"]
N_BOOTSTRAP = 10_000
RNG_SEED = 42


def format_x_labels(levels: list[str], comment_counts: list[int | str]) -> list[str]:
    labels: list[str] = []
    for level, count in zip(levels, comment_counts):
        count_label = "Full" if isinstance(count, str) else f"{count} comments"
        labels.append(f"{level}\n{count_label}")
    return labels


def load_reddit_correctness(project_root: Path) -> pd.DataFrame:
    path = project_root / "results/tables/pool_en_reason_predictions_clean.csv"
    df = pd.read_csv(path)
    df["is_correct"] = (
        df["predicted_candidate_user_id"].astype(str) == df["query_user_id"].astype(str)
    ).astype(int)
    return df.rename(columns={"T_level": "level"})


def load_hn_correctness(project_root: Path) -> pd.DataFrame:
    path = project_root / "results/tables/hn_reason_predictions.csv"
    df = pd.read_csv(path)
    df = df[df["status"] == "success"].copy()
    return df[["level", "query_user_id", "is_correct"]]


def bootstrap_accuracy_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    samples = rng.choice(values, size=(n_bootstrap, n), replace=True)
    means = samples.mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_bootstrap_rows(
    dataset: str,
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level in LEVELS:
        subset = df[df["level"] == level]
        values = subset["is_correct"].to_numpy(dtype=float)
        accuracy, ci_lower, ci_upper = bootstrap_accuracy_ci(values, N_BOOTSTRAP, rng)
        rows.append(
            {
                "dataset": dataset,
                "level": level,
                "accuracy": round(100.0 * accuracy, 2),
                "ci_lower": round(100.0 * ci_lower, 2),
                "ci_upper": round(100.0 * ci_upper, 2),
            }
        )
    return rows


def plot_accuracy_with_ci(rows: list[dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    reddit = [r for r in rows if r["dataset"] == "Reddit"]
    hn = [r for r in rows if r["dataset"] == "Hacker News"]

    x = np.arange(len(LEVELS))
    x_labels = format_x_labels(LEVELS, COMMENT_COUNTS)

    reddit_acc = [float(r["accuracy"]) for r in reddit]
    reddit_lo = [float(r["ci_lower"]) for r in reddit]
    reddit_hi = [float(r["ci_upper"]) for r in reddit]

    hn_acc = [float(r["accuracy"]) for r in hn]
    hn_lo = [float(r["ci_lower"]) for r in hn]
    hn_hi = [float(r["ci_upper"]) for r in hn]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, reddit_acc, marker="o", linewidth=2, label="Reddit / POOL-EN")
    ax.fill_between(x, reddit_lo, reddit_hi, alpha=0.2)
    ax.plot(x, hn_acc, marker="s", linewidth=2, label="Hacker News")
    ax.fill_between(x, hn_lo, hn_hi, alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Deanonymization Accuracy vs Text Volume (95% Bootstrap CI)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ymax = max(max(reddit_hi), max(hn_hi)) * 1.1
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "results/figures"
    tables_dir = project_root / "results/tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_csv = tables_dir / "bootstrap_confidence_intervals.csv"
    out_png = figures_dir / "reddit_vs_hn_accuracy_with_ci.png"
    out_pdf = figures_dir / "reddit_vs_hn_accuracy_with_ci.pdf"

    rng = np.random.default_rng(RNG_SEED)
    reddit_df = load_reddit_correctness(project_root)
    hn_df = load_hn_correctness(project_root)

    rows = compute_bootstrap_rows("Reddit", reddit_df, rng)
    rows.extend(compute_bootstrap_rows("Hacker News", hn_df, rng))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "level", "accuracy", "ci_lower", "ci_upper"],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_accuracy_with_ci(rows, out_png, out_pdf)

    print("Saved outputs:")
    print(f"- {out_csv}")
    print(f"- {out_png}")
    print(f"- {out_pdf}")
    print("\nBootstrap confidence intervals:")
    for row in rows:
        print(
            f"{row['dataset']} {row['level']}: "
            f"{row['accuracy']:.2f}% [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
        )


if __name__ == "__main__":
    main()
