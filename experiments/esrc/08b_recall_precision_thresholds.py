from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
LEVELS_ORDER = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
PRECISION_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
PROTECTED_FILES = ['results/tables/hn_recall_at_precision.csv', 'results/tables/pool_en_recall_at_precision.csv', 'results/tables/hn_precision_recall_curve.csv', 'results/tables/pool_en_precision_recall_curve.csv', 'paper.tex']

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    predictions_csv: Path
    long_csv: Path
    wide_csv: Path
    figure_path: Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Recall at multiple precision thresholds by truncation level.')
    p.add_argument('--dataset', choices=['reddit', 'hn', 'both'], default='both', help='Dataset(s) to analyze (default: both).')
    return p.parse_args()

def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> float:
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return float('nan')
    (precision, recall, _) = precision_recall_curve(y_true, y_score)
    eligible = precision >= min_precision
    if not np.any(eligible):
        return float('nan')
    return float(np.max(recall[eligible]))

def load_reddit_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    level_col = 'T_level' if 'T_level' in df.columns else 'level'
    required = {'query_user_id', 'predicted_candidate_user_id', 'confidence', 'status', level_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Reddit predictions missing columns: {sorted(missing)}')
    df = df[df['status'].astype(str).isin(['ok', 'success'])].copy()
    if df.empty:
        raise ValueError('No successful Reddit predictions found (status ok/success).')
    df['T_level'] = df[level_col].astype(str)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    df['is_correct'] = (df['query_user_id'].astype(str) == df['predicted_candidate_user_id'].astype(str)).astype(int)
    return df[['T_level', 'confidence', 'is_correct']]

def load_hn_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'level', 'query_user_id', 'predicted_candidate_id', 'true_candidate_id', 'confidence', 'is_correct', 'status'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'HN predictions missing columns: {sorted(missing)}')
    df = df[df['status'].astype(str) == 'success'].copy()
    if df.empty:
        raise ValueError('No successful HN predictions found (status == "success").')
    df['T_level'] = df['level'].astype(str)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    df['is_correct'] = pd.to_numeric(df['is_correct'], errors='coerce').fillna(0).astype(int)
    return df[['T_level', 'confidence', 'is_correct']]

def threshold_label(threshold: float) -> str:
    return f'recall_at_{int(round(100 * threshold))}_precision'

def compute_level_thresholds(level_df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    y_true = level_df['is_correct'].to_numpy(dtype=int)
    y_score = level_df['confidence'].to_numpy(dtype=float)
    n_total = int(len(level_df))
    n_correct = int(y_true.sum())
    long_rows: list[dict[str, object]] = []
    wide_row: dict[str, object] = {'n_total': n_total, 'n_correct': n_correct}
    for threshold in PRECISION_THRESHOLDS:
        recall = recall_at_precision(y_true, y_score, threshold)
        long_rows.append({'precision_threshold': threshold, 'precision_threshold_pct': int(round(100 * threshold)), 'recall': recall})
        wide_row[threshold_label(threshold)] = recall
    return (long_rows, wide_row)

def plot_dataset(config: DatasetConfig, long_df: pd.DataFrame) -> None:
    config.figure_path.parent.mkdir(parents=True, exist_ok=True)
    (fig, ax) = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('viridis', len(LEVELS_ORDER))
    for (i, level) in enumerate(LEVELS_ORDER):
        sub = long_df[long_df['T_level'] == level].sort_values('precision_threshold')
        if sub.empty:
            continue
        x = sub['precision_threshold_pct'].astype(float)
        y = sub['recall'].astype(float).fillna(0.0) * 100.0
        ax.plot(x, y, marker='o', linewidth=1.8, label=level, color=cmap(i))
    ax.set_xlabel('Precision threshold (%)')
    ax.set_ylabel('Recall (%)')
    ax.set_title(f'Recall vs. Precision Threshold — {config.display_name}')
    ax.set_xticks([int(round(100 * t)) for t in PRECISION_THRESHOLDS])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(title='T-level', ncol=2, fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(config.figure_path, dpi=300, bbox_inches='tight')
    fig.savefig(config.figure_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def process_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.name == 'reddit':
        df = load_reddit_predictions(config.predictions_csv)
    else:
        df = load_hn_predictions(config.predictions_csv)
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    print(f'\n=== {config.display_name.upper()} ===')
    for level in LEVELS_ORDER:
        level_df = df[df['T_level'] == level].copy()
        if level_df.empty:
            continue
        (level_long, level_wide) = compute_level_thresholds(level_df)
        for row in level_long:
            long_rows.append({'dataset': config.name, 'T_level': level, 'n_total': level_wide['n_total'], 'n_correct': level_wide['n_correct'], **row})
        wide_rows.append({'dataset': config.name, 'T_level': level, **level_wide})
        print(f"\n{level} (n={level_wide['n_total']}, correct={level_wide['n_correct']}):")
        for threshold in PRECISION_THRESHOLDS:
            key = threshold_label(threshold)
            value = level_wide[key]
            label = 'N/A' if pd.isna(value) else f'{100.0 * float(value):.2f}%'
            print(f'  Recall@{int(round(100 * threshold))}% Precision: {label}')
    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    config.long_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(config.long_csv, index=False)
    wide_df.to_csv(config.wide_csv, index=False)
    plot_dataset(config, long_df)
    return (long_df, wide_df)

def verify_protected_files(root: Path, before_mtimes: dict[str, float]) -> None:
    for rel in PROTECTED_FILES:
        path = root / rel
        if not path.exists():
            continue
        if rel in before_mtimes and path.stat().st_mtime != before_mtimes[rel]:
            raise RuntimeError(f'Protected file was modified: {rel}')

def main() -> None:
    args = parse_args()
    root = project_root()
    before_mtimes = {rel: (root / rel).stat().st_mtime for rel in PROTECTED_FILES if (root / rel).exists()}
    configs = [DatasetConfig(name='reddit', display_name='Reddit', predictions_csv=root / 'results/tables/pool_en_reason_predictions_clean.csv', long_csv=root / 'results/tables/pool_en_recall_precision_thresholds.csv', wide_csv=root / 'results/tables/pool_en_recall_precision_thresholds_wide.csv', figure_path=root / 'results/figures/recall_vs_precision_threshold_reddit.png'), DatasetConfig(name='hn', display_name='Hacker News', predictions_csv=root / 'results/tables/hn_reason_predictions.csv', long_csv=root / 'results/tables/hn_recall_precision_thresholds.csv', wide_csv=root / 'results/tables/hn_recall_precision_thresholds_wide.csv', figure_path=root / 'results/figures/recall_vs_precision_threshold_hn.png')]
    selected = {args.dataset} if args.dataset != 'both' else {'reddit', 'hn'}
    combined_long: list[pd.DataFrame] = []
    combined_wide: list[pd.DataFrame] = []
    for config in configs:
        if config.name not in selected:
            continue
        (long_df, wide_df) = process_dataset(config)
        combined_long.append(long_df)
        combined_wide.append(wide_df)
    if combined_long:
        combined_long_path = root / 'results/tables/recall_precision_thresholds.csv'
        combined_wide_path = root / 'results/tables/recall_precision_thresholds_wide.csv'
        pd.concat(combined_long, ignore_index=True).to_csv(combined_long_path, index=False)
        pd.concat(combined_wide, ignore_index=True).to_csv(combined_wide_path, index=False)
    verify_protected_files(root, before_mtimes)
    print('\nWrote:')
    for config in configs:
        if config.name not in selected:
            continue
        print(f'  - {config.long_csv}')
        print(f'  - {config.wide_csv}')
        print(f'  - {config.figure_path}')
    if combined_long:
        print(f'  - {combined_long_path}')
        print(f'  - {combined_wide_path}')
if __name__ == '__main__':
    main()
