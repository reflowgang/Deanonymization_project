from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt


LEVELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
COMMENT_COUNTS = [5, 10, 25, 50, 100, 200, 500, 500]
PENALTIES = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

REDDIT_ACCURACY = [8.8, 13.4, 16.6, 25.2, 28.2, 33.2, 39.8, 39.4]
HN_ACCURACY = [4.8, 8.6, 15.6, 22.2, 26.0, 32.4, 34.6, 38.0]


def filter_breakpoints(bkps: list[int], n_samples: int) -> list[int]:
    return [int(b) for b in bkps if int(b) < n_samples]


def detect_breakpoints(accuracy: list[float], pen: float) -> list[int]:
    signal = np.array(accuracy, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal)
    bkps = algo.predict(pen=pen)
    return filter_breakpoints(list(bkps), len(accuracy))


def breakpoints_to_levels(bkps: list[int]) -> str:
    if not bkps:
        return "none"
    return ";".join(LEVELS[idx] for idx in bkps)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "results/figures"
    tables_dir = project_root / "results/tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Reddit": REDDIT_ACCURACY,
        "Hacker News": HN_ACCURACY,
    }

    rows: list[dict[str, object]] = []
    for dataset_name, accuracy in datasets.items():
        for pen in PENALTIES:
            bkps = detect_breakpoints(accuracy, pen=pen)
            rows.append(
                {
                    "dataset": dataset_name,
                    "penalty": pen,
                    "breakpoints": breakpoints_to_levels(bkps),
                }
            )

    out_csv = tables_dir / "mst_penalty_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(LEVELS))
    x_labels = [f"{lvl}\n{c}" for lvl, c in zip(LEVELS, COMMENT_COUNTS)]

    for ax, (dataset_name, accuracy) in zip(axes, datasets.items()):
        ax.plot(x, accuracy, marker="o", linewidth=2, color="C0")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(dataset_name)
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.4)

        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(PENALTIES)))
        for pen, color in zip(PENALTIES, colors):
            bkps = detect_breakpoints(accuracy, pen=pen)
            for cp in bkps:
                ax.axvline(cp, color=color, linestyle=":", alpha=0.55, linewidth=1.2)
            label = f"pen={pen:g}: {breakpoints_to_levels(bkps)}"
            ax.plot([], [], color=color, linestyle=":", linewidth=1.2, label=label)

        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("MST Penalty Sensitivity (PELT, RBF)")
    fig.tight_layout()

    out_png = figures_dir / "mst_penalty_sensitivity.png"
    out_pdf = figures_dir / "mst_penalty_sensitivity.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved outputs:")
    print(f"- {out_csv}")
    print(f"- {out_png}")
    print(f"- {out_pdf}")
    print("\nMST penalty sensitivity:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
