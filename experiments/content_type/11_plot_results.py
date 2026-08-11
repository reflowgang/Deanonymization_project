from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import paths, type_folders

TYPE_DISPLAY = {
    "personal": "Personal",
    "opinion": "Opinion",
    "topical": "Topical",
}


def main() -> None:
    p = paths()
    recall_csv = p["recall_out"]
    if not recall_csv.exists():
        raise FileNotFoundError(f"Summary not found: {recall_csv}. Run 09_calibrate_precision_recall.py.")

    summary = pd.read_csv(recall_csv)
    summary = summary.set_index("content_type").reindex(list(type_folders())).reset_index()

    p["figures_dir"].mkdir(parents=True, exist_ok=True)

    labels = [TYPE_DISPLAY.get(ct, str(ct).capitalize()) for ct in summary["content_type"]]
    top1 = 100.0 * summary["top1_accuracy"].astype(float)
    recall90 = 100.0 * summary["recall_at_90_precision"].astype(float)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(labels, top1, color=["#4C72B0", "#55A868", "#C44E52"])
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("Content-Type Experiment: Top-1 Accuracy by Type")
    ax1.set_ylim(0, max(top1.max() * 1.15, 1))
    for i, v in enumerate(top1):
        ax1.text(i, float(v) + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    fig1.tight_layout()
    out_top1 = p["figures_dir"] / "content_type_top1_accuracy.png"
    fig1.savefig(out_top1, dpi=300, bbox_inches="tight")
    fig1.savefig(out_top1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    recall90_plot = recall90.fillna(0.0)
    ax2.bar(labels, recall90_plot, color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.set_ylabel("Recall@90% Precision (%)")
    ax2.set_title("Content-Type Experiment: Recall@90% Precision by Type")
    ax2.set_ylim(0, max(recall90_plot.max() * 1.25, 1))
    for i, v in enumerate(recall90_plot):
        label = "N/A" if pd.isna(recall90.iloc[i]) else f"{v:.2f}%"
        ax2.text(i, float(v) + 0.05, label, ha="center", va="bottom", fontsize=10)
    fig2.tight_layout()
    out_recall = p["figures_dir"] / "content_type_recall_at_90.png"
    fig2.savefig(out_recall, dpi=300, bbox_inches="tight")
    fig2.savefig(out_recall.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)

    print("Saved figures:")
    print(f"- {out_top1}")
    print(f"- {out_recall}")


if __name__ == "__main__":
    main()
