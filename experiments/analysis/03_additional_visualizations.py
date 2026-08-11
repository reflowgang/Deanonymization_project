from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
COMMENT_COUNTS = [5, 10, 25, 50, 100, 200, 500, 500]

REDDIT_ACCURACY = np.array([8.8, 13.4, 16.6, 25.2, 28.2, 33.2, 39.8, 39.4], dtype=float)
HN_ACCURACY = np.array([4.8, 8.6, 15.6, 22.2, 26.0, 32.4, 34.6, 38.0], dtype=float)
HN_FAISS_RECALL_AT_15 = np.array([16.8, 19.6, 32.0, 39.6, 45.4, 52.8, 57.6, 59.4], dtype=float)

HN_CALIB_CONFIDENCE = np.array([0.10, 0.20, 0.30, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], dtype=float)
HN_CALIB_EMPIRICAL = np.array(
    [0.0, 0.0, 0.0, 0.0, 9.6774, 9.1837, 14.4126, 26.6513, 49.0050, 100.0],
    dtype=float,
)
HN_CALIB_COUNTS = np.array([2, 4, 3, 4, 279, 98, 1464, 1741, 402, 3], dtype=int)

TRANSITIONS = [f"{LEVELS[i]}→{LEVELS[i + 1]}" for i in range(len(LEVELS) - 1)]


def format_level_labels() -> list[str]:
    labels: list[str] = []
    for level, count in zip(LEVELS, COMMENT_COUNTS):
        labels.append(f"{level}\n{count} comments")
    return labels


def save_figure(fig: plt.Figure, base_path: Path) -> list[Path]:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def build_tables() -> dict[str, pd.DataFrame]:
    reddit_gain = np.diff(REDDIT_ACCURACY)
    hn_gain = np.diff(HN_ACCURACY)

    gain_df = pd.DataFrame(
        {
            "transition": TRANSITIONS,
            "reddit_gain": reddit_gain,
            "hn_gain": hn_gain,
        }
    )

    reddit_relative = 100.0 * REDDIT_ACCURACY / REDDIT_ACCURACY[-1]
    hn_relative = 100.0 * HN_ACCURACY / HN_ACCURACY[-1]

    relative_df = pd.DataFrame(
        {
            "level": LEVELS,
            "comment_count": COMMENT_COUNTS,
            "reddit_relative": reddit_relative,
            "hn_relative": hn_relative,
        }
    )

    calib_df = pd.DataFrame(
        {
            "confidence": HN_CALIB_CONFIDENCE,
            "empirical_accuracy": HN_CALIB_EMPIRICAL,
            "count": HN_CALIB_COUNTS,
        }
    )

    search_reason_df = pd.DataFrame(
        {
            "level": LEVELS,
            "comment_count": COMMENT_COUNTS,
            "faiss_recall_at_15": HN_FAISS_RECALL_AT_15,
            "gpt4o_top1_accuracy": HN_ACCURACY,
        }
    )

    gap_df = pd.DataFrame(
        {
            "level": LEVELS,
            "comment_count": COMMENT_COUNTS,
            "reddit_accuracy": REDDIT_ACCURACY,
            "hn_accuracy": HN_ACCURACY,
            "reddit_minus_hn": REDDIT_ACCURACY - HN_ACCURACY,
        }
    )

    return {
        "accuracy_gain_per_step": gain_df,
        "relative_deanonymization_power": relative_df,
        "hn_confidence_calibration": calib_df,
        "search_vs_reason_hn": search_reason_df,
        "reddit_hn_accuracy_gap": gap_df,
    }


def plot_accuracy_gain_per_step() -> plt.Figure:
    x = np.arange(len(TRANSITIONS))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, np.diff(REDDIT_ACCURACY), marker="o", linewidth=2, label="Reddit / POOL-EN")
    ax.plot(x, np.diff(HN_ACCURACY), marker="s", linewidth=2, label="Hacker News")
    ax.set_xticks(x)
    ax.set_xticklabels(TRANSITIONS, rotation=45, ha="right")
    ax.set_ylabel("Accuracy gain (percentage points)")
    ax.set_title("Accuracy Gain Between Consecutive Truncation Levels")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
    return fig


def plot_relative_deanonymization_power() -> plt.Figure:
    x = np.arange(len(LEVELS))
    reddit_relative = 100.0 * REDDIT_ACCURACY / REDDIT_ACCURACY[-1]
    hn_relative = 100.0 * HN_ACCURACY / HN_ACCURACY[-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, reddit_relative, marker="o", linewidth=2, label="Reddit / POOL-EN")
    ax.plot(x, hn_relative, marker="s", linewidth=2, label="Hacker News")
    ax.set_xticks(x)
    ax.set_xticklabels(format_level_labels())
    ax.set_ylabel("Relative deanonymization power (% of T8)")
    ax.set_title("Relative Deanonymization Power vs Text Volume")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)
    return fig


def plot_confidence_calibration_hn() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = 20 + 200 * (HN_CALIB_COUNTS / HN_CALIB_COUNTS.max())
    ax.scatter(
        HN_CALIB_CONFIDENCE,
        HN_CALIB_EMPIRICAL,
        s=sizes,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
        label="Empirical accuracy",
    )
    ref_x = np.linspace(0, 1, 100)
    ax.plot(ref_x, ref_x * 100.0, linestyle="--", color="gray", label="Perfect calibration (y = x)")
    ax.set_xlabel("GPT-4o confidence")
    ax.set_ylabel("Empirical accuracy (%)")
    ax.set_title("Hacker News GPT-4o Confidence Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def plot_search_vs_reason_hn() -> plt.Figure:
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x,
        HN_FAISS_RECALL_AT_15,
        marker="o",
        linewidth=2,
        label="Search stage: FAISS Recall@15",
    )
    ax.plot(
        x,
        HN_ACCURACY,
        marker="s",
        linewidth=2,
        label="Reason stage: GPT-4o Top-1 Accuracy",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(format_level_labels())
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Hacker News: Search vs Reason Performance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def plot_reddit_hn_accuracy_gap() -> plt.Figure:
    gap = REDDIT_ACCURACY - HN_ACCURACY
    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, gap, marker="o", linewidth=2, color="tab:purple", label="Reddit − Hacker News")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(format_level_labels())
    ax.set_ylabel("Accuracy gap (percentage points)")
    ax.set_title("Reddit vs Hacker News Accuracy Gap")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "results/figures"
    tables_dir = project_root / "results/tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    tables = build_tables()

    table_files = {
        "accuracy_gain_per_step": tables_dir / "accuracy_gain_per_step.csv",
        "relative_deanonymization_power": tables_dir / "relative_deanonymization_power.csv",
        "hn_confidence_calibration": tables_dir / "hn_confidence_calibration.csv",
        "search_vs_reason_hn": tables_dir / "search_vs_reason_hn.csv",
        "reddit_hn_accuracy_gap": tables_dir / "reddit_hn_accuracy_gap.csv",
    }
    for key, path in table_files.items():
        tables[key].to_csv(path, index=False)
        saved_paths.append(path)

    figure_specs = [
        ("accuracy_gain_per_step", plot_accuracy_gain_per_step),
        ("relative_deanonymization_power", plot_relative_deanonymization_power),
        ("confidence_calibration_hn", plot_confidence_calibration_hn),
        ("search_vs_reason_hn", plot_search_vs_reason_hn),
        ("reddit_hn_accuracy_gap", plot_reddit_hn_accuracy_gap),
    ]

    for stem, plot_fn in figure_specs:
        fig = plot_fn()
        fig.tight_layout()
        saved_paths.extend(save_figure(fig, figures_dir / stem))

    print("Saved outputs:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
