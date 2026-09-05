from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
LEVELS_ORDER: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']

@dataclass(frozen=True)
class Paths:
    predictions_csv: Path
    out_figure_with_ci: Path
    out_table_csv: Path

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(predictions_csv=project_root / 'results/tables/pool_en_reason_predictions.csv', out_figure_with_ci=project_root / 'results/figures/pool_en_accuracy_curve_with_ci.png', out_table_csv=project_root / 'results/tables/pool_en_accuracy_by_truncation.csv')
    if not paths.predictions_csv.exists():
        raise FileNotFoundError(f'Predictions CSV not found: {paths.predictions_csv}')
    df = pd.read_csv(paths.predictions_csv)
    required = {'T_level', 'query_user_id', 'predicted_candidate_user_id'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Predictions CSV missing columns: {sorted(missing)}')
    df['T_level'] = df['T_level'].astype(str)
    df['query_user_id'] = df['query_user_id'].astype(str)
    df['predicted_candidate_user_id'] = df['predicted_candidate_user_id'].astype(str)
    df['correct'] = (df['query_user_id'] == df['predicted_candidate_user_id']).astype(int)
    summary_rows: list[dict[str, object]] = []
    for lvl in LEVELS_ORDER:
        sub = df[df['T_level'] == lvl]
        n = int(len(sub))
        correct = int(sub['correct'].sum()) if n else 0
        accuracy = correct / n if n else float('nan')
        if n and np.isfinite(accuracy):
            standard_error = float(np.sqrt(accuracy * (1.0 - accuracy) / n))
            ci95 = 1.96 * standard_error
            ci_low = max(0.0, accuracy - ci95)
            ci_high = min(1.0, accuracy + ci95)
        else:
            standard_error = float('nan')
            ci_low = float('nan')
            ci_high = float('nan')
        summary_rows.append({'T_level': lvl, 'n': n, 'correct': correct, 'accuracy': float(accuracy) if np.isfinite(accuracy) else float('nan'), 'standard_error': standard_error, 'ci_low': ci_low, 'ci_high': ci_high})
    summary_df = pd.DataFrame(summary_rows)
    _ensure_parent_dir(paths.out_table_csv)
    summary_df.to_csv(paths.out_table_csv, index=False)
    print('\nPOOL-EN accuracy by truncation level (with 95% CI)')
    print('T_level\tn\tcorrect\taccuracy\tstandard_error\tci_low\tci_high')
    for (_, r) in summary_df.iterrows():
        lvl = str(r['T_level'])
        n = int(r['n'])
        correct = int(r['correct'])
        acc = float(r['accuracy'])
        se = float(r['standard_error'])
        lo = float(r['ci_low'])
        hi = float(r['ci_high'])
        print(f'{lvl}\t{n}\t{correct}\t{acc:.4f}\t{se:.6f}\t{lo:.4f}\t{hi:.4f}')
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise RuntimeError('matplotlib is required but not installed. Install it, e.g.\n  pip install matplotlib\nThen rerun this script.') from e
    _ensure_parent_dir(paths.out_figure_with_ci)
    xs = LEVELS_ORDER
    ys = summary_df['accuracy'].to_numpy(dtype=float)
    ci_low = summary_df['ci_low'].to_numpy(dtype=float)
    ci_high = summary_df['ci_high'].to_numpy(dtype=float)
    yerr = np.vstack([ys - ci_low, ci_high - ys])
    plt.figure(figsize=(9, 5))
    plt.errorbar(xs, ys, yerr=yerr, fmt='-o', linewidth=2, capsize=4)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel('Number of Comments (T1–T8)')
    plt.ylabel('Deanonymization Accuracy')
    plt.title('Accuracy vs Text Volume with 95% CI')
    plt.tight_layout()
    plt.savefig(paths.out_figure_with_ci, dpi=200)
    plt.close()
    print(f'\nWrote table CSV: {paths.out_table_csv}')
    print(f'Saved figure: {paths.out_figure_with_ci}')
if __name__ == '__main__':
    main()
