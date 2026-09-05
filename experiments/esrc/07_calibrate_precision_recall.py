from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
LEVELS_ORDER: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
CURVE_COLUMNS = ['T_level', 'threshold', 'predicted_positive', 'true_positive', 'false_positive', 'false_negative', 'precision', 'recall']
SUMMARY_COLUMNS = ['T_level', 'n_total', 'n_correct_total', 'top1_accuracy', 'recall_at_90_precision', 'recall_at_99_precision']

@dataclass(frozen=True)
class Paths:
    predictions_csv: Path
    curve_csv: Path
    summary_csv: Path

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def recall_at_precision(curve_df: pd.DataFrame, min_precision: float) -> float:
    if curve_df.empty:
        return float('nan')
    eligible = curve_df[curve_df['precision'] >= min_precision]
    if eligible.empty:
        return float('nan')
    return float(eligible['recall'].max())

def compute_pr_curve_for_level(level: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sub = sub.sort_values('confidence', ascending=False, kind='mergesort').reset_index(drop=True)
    conf = sub['confidence'].to_numpy(dtype=float)
    correct = sub['correct'].to_numpy(dtype=int)
    n_total = int(len(sub))
    n_correct_total = int(correct.sum())
    top1_accuracy = n_correct_total / n_total if n_total else float('nan')
    curve_rows: list[dict[str, object]] = []
    for i in range(n_total):
        threshold = float(conf[i])
        mask = conf >= threshold
        predicted_positive = int(mask.sum())
        true_positive = int(correct[mask].sum())
        false_positive = predicted_positive - true_positive
        false_negative = n_correct_total - true_positive
        precision = true_positive / predicted_positive if predicted_positive else float('nan')
        recall = true_positive / n_correct_total if n_correct_total else float('nan')
        curve_rows.append({'T_level': level, 'threshold': threshold, 'predicted_positive': predicted_positive, 'true_positive': true_positive, 'false_positive': false_positive, 'false_negative': false_negative, 'precision': precision, 'recall': recall})
    curve_df = pd.DataFrame(curve_rows)
    summary = {'T_level': level, 'n_total': n_total, 'n_correct_total': n_correct_total, 'top1_accuracy': float(top1_accuracy), 'recall_at_90_precision': recall_at_precision(curve_df, 0.9), 'recall_at_99_precision': recall_at_precision(curve_df, 0.99)}
    return (curve_df, summary)

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(predictions_csv=project_root / 'results/tables/pool_en_reason_predictions_clean.csv', curve_csv=project_root / 'results/tables/pool_en_precision_recall_curve.csv', summary_csv=project_root / 'results/tables/pool_en_recall_at_precision.csv')
    if not paths.predictions_csv.exists():
        raise FileNotFoundError(f'Predictions CSV not found: {paths.predictions_csv}')
    df = pd.read_csv(paths.predictions_csv)
    required = {'T_level', 'query_user_id', 'predicted_candidate_user_id', 'confidence', 'status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Predictions CSV missing columns: {sorted(missing)}')
    df['T_level'] = df['T_level'].astype(str)
    df['query_user_id'] = df['query_user_id'].astype(str)
    df['predicted_candidate_user_id'] = df['predicted_candidate_user_id'].astype(str)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    df = df[df['status'].astype(str) == 'ok'].copy()
    if df.empty:
        raise ValueError('No rows with status == "ok" in predictions file.')
    df['correct'] = (df['query_user_id'] == df['predicted_candidate_user_id']).astype(int)
    curve_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    levels_present = set(df['T_level'].unique())
    for level in LEVELS_ORDER:
        if level not in levels_present:
            continue
        sub = df[df['T_level'] == level].copy()
        (curve_df, summary) = compute_pr_curve_for_level(level, sub)
        curve_parts.append(curve_df)
        summary_rows.append(summary)
    extra_levels = sorted(levels_present - set(LEVELS_ORDER))
    for level in extra_levels:
        sub = df[df['T_level'] == level].copy()
        (curve_df, summary) = compute_pr_curve_for_level(level, sub)
        curve_parts.append(curve_df)
        summary_rows.append(summary)
    curve_out = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame(columns=CURVE_COLUMNS)
    summary_out = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    _ensure_parent_dir(paths.curve_csv)
    _ensure_parent_dir(paths.summary_csv)
    curve_out.to_csv(paths.curve_csv, index=False)
    summary_out.to_csv(paths.summary_csv, index=False)
    print('\nPOOL-EN precision-recall calibration (status == ok)')
    print('T_level\tn_total\tn_correct_total\ttop1_accuracy\trecall_at_90_precision\trecall_at_99_precision')
    for (_, row) in summary_out.iterrows():
        print(f"{row['T_level']}\t{int(row['n_total'])}\t{int(row['n_correct_total'])}\t{row['top1_accuracy']:.4f}\t{row['recall_at_90_precision']:.4f}\t{row['recall_at_99_precision']:.4f}")
    print(f'\nWrote curve points: {paths.curve_csv}')
    print(f'Wrote summary: {paths.summary_csv}')
if __name__ == '__main__':
    main()
