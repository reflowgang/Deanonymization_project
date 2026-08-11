from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TYPE_LABELS, paths, type_folders

CURVE_COLUMNS = [
    "content_type",
    "threshold",
    "predicted_positive",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
]

SUMMARY_COLUMNS = [
    "content_type",
    "n_total",
    "n_correct_total",
    "top1_accuracy",
    "recall_at_90_precision",
    "recall_at_99_precision",
]


def recall_at_precision(curve_df: pd.DataFrame, min_precision: float) -> float:
    if curve_df.empty:
        return float("nan")
    eligible = curve_df[curve_df["precision"] >= min_precision]
    if eligible.empty:
        return float("nan")
    return float(eligible["recall"].max())


def compute_pr_curve_for_type(content_type: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sub = sub.sort_values("confidence", ascending=False, kind="mergesort").reset_index(drop=True)
    conf = sub["confidence"].to_numpy(dtype=float)
    correct = sub["correct"].to_numpy(dtype=int)

    n_total = int(len(sub))
    n_correct_total = int(correct.sum())
    top1_accuracy = (n_correct_total / n_total) if n_total else float("nan")

    curve_rows: list[dict[str, object]] = []
    for i in range(n_total):
        threshold = float(conf[i])
        mask = conf >= threshold
        predicted_positive = int(mask.sum())
        true_positive = int(correct[mask].sum())
        false_positive = predicted_positive - true_positive
        false_negative = n_correct_total - true_positive
        precision = (
            true_positive / predicted_positive if predicted_positive else float("nan")
        )
        recall = true_positive / n_correct_total if n_correct_total else float("nan")
        curve_rows.append(
            {
                "content_type": content_type,
                "threshold": threshold,
                "predicted_positive": predicted_positive,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    summary = {
        "content_type": content_type,
        "n_total": n_total,
        "n_correct_total": n_correct_total,
        "top1_accuracy": float(top1_accuracy),
        "recall_at_90_precision": recall_at_precision(curve_df, 0.90),
        "recall_at_99_precision": recall_at_precision(curve_df, 0.99),
    }
    return curve_df, summary


def main() -> None:
    p = paths()
    if not p["reason_out"].exists():
        raise FileNotFoundError(f"Predictions not found: {p['reason_out']}. Run 08_reason_top15.py.")

    df = pd.read_csv(p["reason_out"])
    required = {
        "content_type",
        "query_user_id",
        "predicted_candidate_user_id",
        "confidence",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    df = df[df["status"].astype(str) == "ok"].copy()
    if df.empty:
        raise ValueError('No rows with status == "ok".')

    df["query_user_id"] = df["query_user_id"].astype(str)
    df["predicted_candidate_user_id"] = df["predicted_candidate_user_id"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["correct"] = (df["query_user_id"] == df["predicted_candidate_user_id"]).astype(int)

    curve_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for folder in type_folders():
        sub = df[df["content_type"].astype(str) == folder].copy()
        if sub.empty:
            print(f"- {folder}: no successful predictions (skipping)")
            continue
        curve_df, summary = compute_pr_curve_for_type(folder, sub)
        curve_parts.append(curve_df)
        summary_rows.append(summary)

    curve_out = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame(
        columns=CURVE_COLUMNS
    )
    summary_out = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    p["curve_out"].parent.mkdir(parents=True, exist_ok=True)
    curve_out.to_csv(p["curve_out"], index=False)
    summary_out.to_csv(p["recall_out"], index=False)

    esrc_summary = p["summary_out"]
    if esrc_summary.exists():
        qual_summary = pd.read_csv(esrc_summary)
        qual_summary = qual_summary[
            ~qual_summary["metric"].isin(
                {
                    "top1_accuracy_personal",
                    "top1_accuracy_opinion",
                    "top1_accuracy_topical",
                    "recall_at_90_personal",
                    "recall_at_90_opinion",
                    "recall_at_90_topical",
                    "recall_at_99_personal",
                    "recall_at_99_opinion",
                    "recall_at_99_topical",
                }
            )
        ]
        extra_rows = []
        for _, row in summary_out.iterrows():
            ct = str(row["content_type"])
            label = ct[0].upper() if ct else ct
            for folder, key in TYPE_LABELS.items():
                if TYPE_LABELS[folder] == ct:
                    label = folder
                    break
            extra_rows.extend(
                [
                    {"metric": f"top1_accuracy_{ct}", "value": round(100.0 * float(row["top1_accuracy"]), 2)},
                    {
                        "metric": f"recall_at_90_{ct}",
                        "value": round(100.0 * float(row["recall_at_90_precision"]), 2)
                        if pd.notna(row["recall_at_90_precision"])
                        else "",
                    },
                    {
                        "metric": f"recall_at_99_{ct}",
                        "value": round(100.0 * float(row["recall_at_99_precision"]), 2)
                        if pd.notna(row["recall_at_99_precision"])
                        else "",
                    },
                ]
            )
        pd.concat([qual_summary, pd.DataFrame(extra_rows)], ignore_index=True).to_csv(
            esrc_summary, index=False
        )
    else:
        summary_out.to_csv(esrc_summary, index=False)

    print("\nContent-type calibration summary:")
    print(summary_out.to_string(index=False))
    print(f"\nWrote: {p['curve_out']}")
    print(f"Wrote: {p['recall_out']}")
    print(f"Wrote: {p['summary_out']}")


if __name__ == "__main__":
    main()
