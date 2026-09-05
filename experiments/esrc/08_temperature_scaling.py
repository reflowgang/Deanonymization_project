from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, precision_recall_curve
RANDOM_STATE = 42
LEVELS_ORDER = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
PROB_EPS = 1e-06
T_BOUNDS = (0.1, 10.0)
N_ECE_BINS = 10
RESULT_COLUMNS = ['dataset', 'T_level', 'n_total', 'n_correct', 'recall_at_90_original', 'recall_at_90_calibrated_per_level', 'recall_at_90_calibrated_global', 'recall_at_99_original', 'recall_at_99_calibrated_per_level', 'recall_at_99_calibrated_global', 'ece_original', 'ece_calibrated', 'auc_pr_original', 'auc_pr_calibrated', 'delta_recall_90', 'delta_recall_99', 'delta_ece', 'delta_auc_pr', 'temperature_per_level', 'temperature_global']
SUMMARY_COLUMNS = ['dataset', 'weighted_recall_90_original', 'weighted_recall_90_calibrated', 'weighted_recall_99_original', 'weighted_recall_99_calibrated', 'weighted_ece_original', 'weighted_ece_calibrated', 'weighted_auc_pr_original', 'weighted_auc_pr_calibrated', 'global_temperature']
PROTECTED_FILES = ['results/tables/hn_recall_at_precision.csv', 'results/tables/pool_en_recall_at_precision.csv', 'results/tables/hn_precision_recall_curve.csv', 'results/tables/pool_en_precision_recall_curve.csv', 'paper.tex']

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    predictions_csv: Path
    results_csv: Path
    figure_path: Path
    success_status: str
    level_col: str

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Post-hoc temperature scaling calibration (exploratory; writes new files only).')
    p.add_argument('--dataset', choices=['reddit', 'hn', 'both'], default='both', help='Dataset(s) to calibrate (default: both).')
    return p.parse_args()

def prob_to_logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs.astype(float), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(clipped / (1.0 - clipped))

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = prob_to_logit(probs)
    calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
    return np.clip(calibrated, 0.0, 1.0)

def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return 1.0

    def nll(temperature: float) -> float:
        calibrated = apply_temperature(probs, temperature)
        calibrated = np.clip(calibrated, PROB_EPS, 1.0 - PROB_EPS)
        loss = -(labels * np.log(calibrated) + (1.0 - labels) * np.log(1.0 - calibrated))
        return float(np.mean(loss))
    result = minimize_scalar(nll, bounds=T_BOUNDS, method='bounded')
    temperature = float(result.x)
    return float(np.clip(temperature, T_BOUNDS[0], T_BOUNDS[1]))

def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> float:
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return float('nan')
    (precision, recall, _) = precision_recall_curve(y_true, y_score)
    eligible = precision >= min_precision
    if not np.any(eligible):
        return float('nan')
    return float(np.max(recall[eligible]))

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=N_ECE_BINS) -> float:
    if len(y_true) == 0:
        return float('nan')
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        (low, high) = (bins[i], bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += count / n * abs(bin_conf - bin_acc)
    return float(ece)

def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=N_ECE_BINS) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers: list[float] = []
    accuracies: list[float] = []
    for i in range(n_bins):
        (low, high) = (bins[i], bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        centers.append(float(y_prob[mask].mean()))
        accuracies.append(float(y_true[mask].mean()))
    return (np.asarray(centers), np.asarray(accuracies))

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {'recall_at_90': float('nan'), 'recall_at_99': float('nan'), 'ece': float('nan'), 'auc_pr': float('nan')}
    return {'recall_at_90': recall_at_precision(y_true, y_score, 0.9), 'recall_at_99': recall_at_precision(y_true, y_score, 0.99), 'ece': expected_calibration_error(y_true, y_score), 'auc_pr': float(average_precision_score(y_true, y_score)) if int(y_true.sum()) > 0 else float('nan')}

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

def split_calibration_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = len(shuffled) // 2
    return (shuffled.iloc[:split_idx].copy(), shuffled.iloc[split_idx:].copy())

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float('nan')
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    if float(w.sum()) == 0.0:
        return float('nan')
    return float(np.average(v, weights=w))

def plot_dataset_figure(config: DatasetConfig, level_rows: list[dict[str, object]], test_original: np.ndarray, test_calibrated_global: np.ndarray, test_labels: np.ndarray) -> None:
    config.figure_path.parent.mkdir(parents=True, exist_ok=True)
    level_df = pd.DataFrame(level_rows).set_index('T_level').reindex(LEVELS_ORDER).reset_index()
    x_labels = LEVELS_ORDER
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    (orig_centers, orig_acc) = reliability_bins(test_labels, test_original)
    (cal_centers, cal_acc) = reliability_bins(test_labels, test_calibrated_global)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Perfect calibration')
    if len(orig_centers):
        ax.plot(orig_centers, orig_acc, 'o-', label='Original', color='#4C72B0')
    if len(cal_centers):
        ax.plot(cal_centers, cal_acc, 's-', label='Calibrated (global)', color='#C44E52')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title(f'Reliability Diagram — {config.display_name}')
    ax.legend(loc='lower right', fontsize=8)
    ax = axes[1]
    recall90_orig = level_df['recall_at_90_original'].astype(float).to_numpy()
    recall90_cal = level_df['recall_at_90_calibrated_global'].astype(float).to_numpy()
    x = np.arange(len(x_labels))
    ax.plot(x, recall90_orig, 'o-', label='Original', color='#4C72B0')
    ax.plot(x, recall90_cal, 's-', label='Calibrated (global)', color='#C44E52')
    ax.axhline(0.05, linestyle='--', color='gray', linewidth=1.0, label='MST threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('Truncation level')
    ax.set_ylabel('Recall@90% Precision')
    ax.set_title('Recall@90% Precision')
    ax.legend(loc='best', fontsize=8)
    ax = axes[2]
    ax.hist(test_original, bins=20, alpha=0.55, density=True, label='Original', color='#4C72B0')
    ax.hist(test_calibrated_global, bins=20, alpha=0.55, density=True, label='Calibrated (global)', color='#C44E52')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Histogram')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(config.figure_path, dpi=300, bbox_inches='tight')
    fig.savefig(config.figure_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def process_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    if config.name == 'reddit':
        df = load_reddit_predictions(config.predictions_csv)
    else:
        df = load_hn_predictions(config.predictions_csv)
    level_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for level in LEVELS_ORDER:
        level_df = df[df['T_level'] == level].copy()
        if level_df.empty:
            continue
        (cal_df, test_df) = split_calibration_test(level_df)
        level_splits[level] = (cal_df, test_df)
    if not level_splits:
        raise ValueError(f'No truncation levels found for dataset {config.name}.')
    pooled_cal_probs: list[np.ndarray] = []
    pooled_cal_labels: list[np.ndarray] = []
    for (cal_df, _) in level_splits.values():
        pooled_cal_probs.append(cal_df['confidence'].to_numpy(dtype=float))
        pooled_cal_labels.append(cal_df['is_correct'].to_numpy(dtype=int))
    global_temperature = fit_temperature(np.concatenate(pooled_cal_probs), np.concatenate(pooled_cal_labels))
    rows: list[dict[str, object]] = []
    pooled_test_probs: list[np.ndarray] = []
    pooled_test_cal_global: list[np.ndarray] = []
    pooled_test_labels: list[np.ndarray] = []
    print(f'\n=== {config.display_name.upper()} ===\n')
    print(f'Global temperature: {global_temperature:.4f}')
    for level in LEVELS_ORDER:
        if level not in level_splits:
            continue
        (cal_df, test_df) = level_splits[level]
        y_cal = cal_df['is_correct'].to_numpy(dtype=int)
        p_cal = cal_df['confidence'].to_numpy(dtype=float)
        y_test = test_df['is_correct'].to_numpy(dtype=int)
        p_test = test_df['confidence'].to_numpy(dtype=float)
        temp_level = fit_temperature(p_cal, y_cal)
        p_test_level = apply_temperature(p_test, temp_level)
        p_test_global = apply_temperature(p_test, global_temperature)
        orig_metrics = compute_metrics(y_test, p_test)
        level_metrics = compute_metrics(y_test, p_test_level)
        global_metrics = compute_metrics(y_test, p_test_global)
        row = {'dataset': config.name, 'T_level': level, 'n_total': int(len(test_df)), 'n_correct': int(y_test.sum()), 'recall_at_90_original': orig_metrics['recall_at_90'], 'recall_at_90_calibrated_per_level': level_metrics['recall_at_90'], 'recall_at_90_calibrated_global': global_metrics['recall_at_90'], 'recall_at_99_original': orig_metrics['recall_at_99'], 'recall_at_99_calibrated_per_level': level_metrics['recall_at_99'], 'recall_at_99_calibrated_global': global_metrics['recall_at_99'], 'ece_original': orig_metrics['ece'], 'ece_calibrated': global_metrics['ece'], 'auc_pr_original': orig_metrics['auc_pr'], 'auc_pr_calibrated': global_metrics['auc_pr'], 'delta_recall_90': global_metrics['recall_at_90'] - orig_metrics['recall_at_90'], 'delta_recall_99': global_metrics['recall_at_99'] - orig_metrics['recall_at_99'], 'delta_ece': global_metrics['ece'] - orig_metrics['ece'], 'delta_auc_pr': global_metrics['auc_pr'] - orig_metrics['auc_pr'], 'temperature_per_level': temp_level, 'temperature_global': global_temperature}
        rows.append(row)
        pooled_test_probs.append(p_test)
        pooled_test_cal_global.append(p_test_global)
        pooled_test_labels.append(y_test)

        def fmt_pct(v: float) -> str:
            return 'N/A' if pd.isna(v) else f'{100.0 * v:.2f}%'
        print(f'{level}:')
        print(f"  Recall@90 before: {fmt_pct(orig_metrics['recall_at_90'])}")
        print(f"  Recall@90 after:  {fmt_pct(global_metrics['recall_at_90'])}")
        print(f"  Recall@99 before: {fmt_pct(orig_metrics['recall_at_99'])}")
        print(f"  Recall@99 after:  {fmt_pct(global_metrics['recall_at_99'])}")
        print(f"  ECE before: {orig_metrics['ece']:.4f}")
        print(f"  ECE after:  {global_metrics['ece']:.4f}")
        print(f"  AUC-PR before: {orig_metrics['auc_pr']:.4f}")
        print(f"  AUC-PR after:  {global_metrics['auc_pr']:.4f}")
    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    config.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.results_csv, index=False)
    test_original = np.concatenate(pooled_test_probs)
    test_calibrated_global = np.concatenate(pooled_test_cal_global)
    test_labels = np.concatenate(pooled_test_labels)
    plot_dataset_figure(config, rows, test_original, test_calibrated_global, test_labels)
    weights = results_df['n_total'].astype(float)
    summary = {'dataset': config.name, 'weighted_recall_90_original': weighted_mean(results_df['recall_at_90_original'], weights), 'weighted_recall_90_calibrated': weighted_mean(results_df['recall_at_90_calibrated_global'], weights), 'weighted_recall_99_original': weighted_mean(results_df['recall_at_99_original'], weights), 'weighted_recall_99_calibrated': weighted_mean(results_df['recall_at_99_calibrated_global'], weights), 'weighted_ece_original': weighted_mean(results_df['ece_original'], weights), 'weighted_ece_calibrated': weighted_mean(results_df['ece_calibrated'], weights), 'weighted_auc_pr_original': weighted_mean(results_df['auc_pr_original'], weights), 'weighted_auc_pr_calibrated': weighted_mean(results_df['auc_pr_calibrated'], weights), 'global_temperature': global_temperature}
    return (results_df, summary)

def verify_protected_files(root: Path, before_mtimes: dict[str, float]) -> None:
    for rel in PROTECTED_FILES:
        path = root / rel
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if rel in before_mtimes and mtime != before_mtimes[rel]:
            raise RuntimeError(f'Protected file was modified: {rel}')

def print_assessment(summaries: list[dict[str, float]]) -> None:
    print('\nCalibration usefulness assessment\n')
    all_recall_improvements_pp: list[float] = []
    for summary in summaries:
        dataset = summary['dataset']
        mean_ece_reduction = summary['weighted_ece_original'] - summary['weighted_ece_calibrated']
        mean_recall_90_inc = 100.0 * (summary['weighted_recall_90_calibrated'] - summary['weighted_recall_90_original'])
        mean_recall_99_inc = 100.0 * (summary['weighted_recall_99_calibrated'] - summary['weighted_recall_99_original'])
        mean_auc_inc = summary['weighted_auc_pr_calibrated'] - summary['weighted_auc_pr_original']
        all_recall_improvements_pp.extend([abs(mean_recall_90_inc), abs(mean_recall_99_inc)])
        print(f'{dataset.upper()}:')
        print(f'  Mean ECE reduction: {mean_ece_reduction:.4f}')
        print(f'  Mean Recall@90 increase: {mean_recall_90_inc:.2f} pp')
        print(f'  Mean Recall@99 increase: {mean_recall_99_inc:.2f} pp')
        print(f'  Mean AUC-PR increase: {mean_auc_inc:.4f}')
    if all((v < 1.0 for v in all_recall_improvements_pp)):
        print('\nTemperature scaling improves probability calibration but has negligible impact on deanonymization performance.')

def main() -> None:
    args = parse_args()
    root = project_root()
    before_mtimes = {rel: (root / rel).stat().st_mtime for rel in PROTECTED_FILES if (root / rel).exists()}
    configs = [DatasetConfig(name='reddit', display_name='Reddit', predictions_csv=root / 'results/tables/pool_en_reason_predictions_clean.csv', results_csv=root / 'results/tables/pool_en_calibration_results.csv', figure_path=root / 'results/figures/calibration_before_after_reddit.png', success_status='ok', level_col='T_level'), DatasetConfig(name='hn', display_name='Hacker News', predictions_csv=root / 'results/tables/hn_reason_predictions.csv', results_csv=root / 'results/tables/hn_calibration_results.csv', figure_path=root / 'results/figures/calibration_before_after_hn.png', success_status='success', level_col='level')]
    selected = {args.dataset} if args.dataset != 'both' else {'reddit', 'hn'}
    summaries: list[dict[str, float]] = []
    created_files: list[Path] = []
    for config in configs:
        if config.name not in selected:
            continue
        (results_df, summary) = process_dataset(config)
        summaries.append(summary)
        created_files.extend([config.results_csv, config.figure_path, config.figure_path.with_suffix('.pdf')])
        temps = results_df['temperature_per_level'].astype(float)
        if ((temps < T_BOUNDS[0]) | (temps > T_BOUNDS[1])).any():
            raise RuntimeError(f'Per-level temperature out of bounds for {config.name}.')
        if not T_BOUNDS[0] <= summary['global_temperature'] <= T_BOUNDS[1]:
            raise RuntimeError(f'Global temperature out of bounds for {config.name}.')
    summary_path = root / 'results/tables/calibration_summary.csv'
    summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(summary_path, index=False)
    created_files.append(summary_path)
    verify_protected_files(root, before_mtimes)
    print('\nCreated files:')
    for path in created_files:
        print(f'  - {path}')
    print('\nRow counts:')
    for config in configs:
        if config.name not in selected:
            continue
        df = pd.read_csv(config.results_csv)
        print(f'  {config.name}: {len(df)} rows in {config.results_csv.name}')
    print(f'  summary: {len(summary_df)} rows in calibration_summary.csv')
    print('\nTemperatures found:')
    for config in configs:
        if config.name not in selected:
            continue
        df = pd.read_csv(config.results_csv)
        print(f"  {config.name} global T: {df['temperature_global'].iloc[0]:.4f}")
        for (_, row) in df.iterrows():
            print(f"    {row['T_level']}: per-level T={row['temperature_per_level']:.4f}")
    print('\nWeighted summary metrics:')
    print(summary_df.to_string(index=False))
    print_assessment(summaries)
if __name__ == '__main__':
    main()
