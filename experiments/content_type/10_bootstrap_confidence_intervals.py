from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_SEED, paths, type_folders

N_BOOTSTRAP = 10_000
ALPHA = 0.05

TYPE_DISPLAY = {
    "personal": "Personal",
    "opinion": "Opinion",
    "topical": "Topical",
}


def bootstrap_accuracy_ci(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    samples = rng.choice(values, size=(n_bootstrap, n), replace=True)
    means = samples.mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> float:
    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 1.0
    z = (p_a - p_b) / se
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


def load_correctness(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    required = {"content_type", "query_user_id", "predicted_candidate_user_id", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    df = df[df["status"].astype(str) == "ok"].copy()
    if df.empty:
        raise ValueError('No rows with status == "ok".')

    df["content_type"] = df["content_type"].astype(str)
    df["query_user_id"] = df["query_user_id"].astype(str)
    df["predicted_candidate_user_id"] = df["predicted_candidate_user_id"].astype(str)
    df["is_correct"] = (
        df["query_user_id"] == df["predicted_candidate_user_id"]
    ).astype(int)
    return df


def plot_top1_with_ci(ci_df: pd.DataFrame, out_png: Path) -> None:
    plot_df = ci_df.set_index("content_type").reindex(list(type_folders())).reset_index()
    labels = [TYPE_DISPLAY.get(ct, ct.capitalize()) for ct in plot_df["content_type"]]
    x = np.arange(len(type_folders()))
    acc = plot_df["accuracy"].astype(float).to_numpy()
    err_lo = acc - plot_df["ci_lower"].astype(float).to_numpy()
    err_hi = plot_df["ci_upper"].astype(float).to_numpy() - acc
    yerr = np.vstack([err_lo, err_hi])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, acc, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.85)
    ax.errorbar(
        x,
        acc,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        capsize=6,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Content-Type Experiment: Top-1 Accuracy with 95% Bootstrap CI")
    ymax = max(plot_df["ci_upper"].astype(float).max() * 1.15, 1.0)
    ax.set_ylim(0, ymax)
    for i, row in plot_df.iterrows():
        ax.text(
            i,
            float(row["ci_upper"]) + 0.5,
            f"{row['accuracy']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = paths()
    predictions_csv = p["reason_out"]
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_csv}")

    df = load_correctness(predictions_csv)
    rng = np.random.default_rng(RANDOM_SEED)

    ci_rows: list[dict[str, object]] = []
    type_stats: dict[str, tuple[int, int]] = {}

    for folder in type_folders():
        sub = df[df["content_type"] == folder]
        if sub.empty:
            raise ValueError(f"No successful predictions for type '{folder}'.")
        values = sub["is_correct"].to_numpy(dtype=float)
        accuracy, ci_lower, ci_upper = bootstrap_accuracy_ci(values, N_BOOTSTRAP, rng)
        n_correct = int(values.sum())
        n_total = int(len(values))
        type_stats[folder] = (n_correct, n_total)
        ci_rows.append(
            {
                "content_type": folder,
                "accuracy": round(100.0 * accuracy, 2),
                "ci_lower": round(100.0 * ci_lower, 2),
                "ci_upper": round(100.0 * ci_upper, 2),
            }
        )

    bootstrap_out = p["bootstrap_ci"]
    with bootstrap_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["content_type", "accuracy", "ci_lower", "ci_upper"],
        )
        writer.writeheader()
        writer.writerows(ci_rows)

    sig_rows: list[dict[str, object]] = []
    for type_a, type_b in combinations(type_folders(), 2):
        success_a, n_a = type_stats[type_a]
        success_b, n_b = type_stats[type_b]
        p_value = two_proportion_z_test(success_a, n_a, success_b, n_b)
        sig_rows.append(
            {
                "type_a": type_a,
                "type_b": type_b,
                "p_value": round(p_value, 6),
                "significant_0_05": int(p_value < ALPHA),
            }
        )

    sig_out = p["significance_tests"]
    with sig_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["type_a", "type_b", "p_value", "significant_0_05"],
        )
        writer.writeheader()
        writer.writerows(sig_rows)

    ci_df = pd.DataFrame(ci_rows)
    fig_out = p["figures_dir"] / "content_type_top1_accuracy_with_ci.png"
    p["figures_dir"].mkdir(parents=True, exist_ok=True)
    plot_top1_with_ci(ci_df, fig_out)

    print("Bootstrap 95% confidence intervals (top-1 accuracy):")
    for row in ci_rows:
        print(
            f"  {row['content_type']}: "
            f"{row['accuracy']:.2f}% [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
        )

    print("\nPairwise two-proportion z-tests:")
    for row in sig_rows:
        sig_label = "yes" if row["significant_0_05"] else "no"
        print(
            f"  {row['type_a']} vs {row['type_b']}: "
            f"p = {row['p_value']:.6f} (significant at 0.05: {sig_label})"
        )

    print("\nSaved outputs:")
    print(f"- {bootstrap_out}")
    print(f"- {sig_out}")
    print(f"- {fig_out}")


if __name__ == "__main__":
    main()
