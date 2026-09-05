from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.metrics_calibration import calibrate_scores, evaluate_at_threshold, evaluate_scores, fit_isotonic, threshold_for_precision
from esrc.paths import summer_root
DEFAULT_SCORES = summer_root() / 'results/runs/p2_regression_50_T8/estimator_scores.csv'
OUT_DIR = summer_root() / 'results/p3_calibration/tables'
CAL_SEED = 42

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P3 isotonic calibration on fixture half-split')
    p.add_argument('--scores-csv', default=str(DEFAULT_SCORES))
    p.add_argument('--out-dir', default=str(OUT_DIR))
    p.add_argument('--seed', type=int, default=CAL_SEED)
    p.add_argument('--estimator', choices=('a', 'b', 'both'), default='both')
    return p.parse_args()

def as_bool(v: str) -> bool:
    return str(v).lower() in {'1', 'true', 'yes'}

def fmt(x: float | None, d: int=4) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and x != x:
        return ''
    return f'{x:.{d}f}'

def half_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mid = n // 2
    return (idx[:mid], idx[mid:])

def run_estimator(name: str, conf: np.ndarray, y: np.ndarray, user_ids: list[str], cal_idx: np.ndarray, test_idx: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    (conf_cal, y_cal) = (conf[cal_idx], y[cal_idx])
    (conf_test, y_test) = (conf[test_idx], y[test_idx])
    ids_test = [user_ids[i] for i in test_idx]
    iso = fit_isotonic(conf_cal, y_cal)
    cal_on_test = calibrate_scores(iso, conf_test)
    thr_raw = threshold_for_precision(conf_cal, y_cal, 0.9)
    thr_cal = threshold_for_precision(calibrate_scores(iso, conf_cal), y_cal, 0.9)
    rows = []
    for (label, scores, thr) in [('naive_raw_0.9', conf_test, 0.9), ('isotonic_then_0.9', cal_on_test, 0.9), ('raw_thr_from_cal', conf_test, thr_raw if thr_raw is not None else 0.9), ('isotonic_thr_from_cal', cal_on_test, thr_cal if thr_cal is not None else 0.9)]:
        ev = evaluate_at_threshold(scores, y_test, thr if thr is not None else 0.9)
        rows.append({'estimator': name, 'policy': label, 'threshold': fmt(ev.threshold), 'n_test': len(test_idx), 'n_accepted': ev.n_accepted, 'n_correct_accepted': ev.n_correct_accepted, 'precision': fmt(ev.precision), 'recall': fmt(ev.recall), 'cal_threshold_learned': fmt(thr) if 'from_cal' in label else ''})
    detail = []
    for (i, uid) in enumerate(ids_test):
        detail.append({'estimator': name, 'query_user_id': uid, 'correct': str(bool(y_test[i])), 'raw_conf': fmt(float(conf_test[i])), 'isotonic_conf': fmt(float(cal_on_test[i])), 'naive_accept': str(bool(conf_test[i] >= 0.9)), 'isotonic_accept': str(bool(cal_on_test[i] >= 0.9))})
    m_raw_test = evaluate_scores(conf_test, y_test)
    m_iso_test = evaluate_scores(cal_on_test, y_test)
    summary = {'estimator': name, 'n_cal': int(len(cal_idx)), 'n_test': int(len(test_idx)), 'cal_n_correct': int(y_cal.sum()), 'test_n_correct': int(y_test.sum()), 'thr_raw_for_p90': thr_raw, 'thr_isotonic_for_p90': thr_cal, 'test_ece_raw': m_raw_test.ece, 'test_ece_isotonic': m_iso_test.ece, 'test_brier_raw': m_raw_test.brier, 'test_brier_isotonic': m_iso_test.brier, 'test_r90_raw': m_raw_test.recall_at_90_precision, 'test_r90_isotonic': m_iso_test.recall_at_90_precision}
    return (rows, detail, summary)

def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = Path(args.scores_csv)
    rows = [r for r in csv.DictReader(scores_path.open(newline='', encoding='utf-8')) if r.get('status') == 'ok']
    y = np.array([as_bool(r['correct']) for r in rows], dtype=bool)
    user_ids = [r['query_user_id'] for r in rows]
    (cal_idx, test_idx) = half_split(len(rows), args.seed)
    estimators = []
    if args.estimator in ('a', 'both'):
        estimators.append(('a_verbalized', np.array([float(r['score_a']) for r in rows])))
    if args.estimator in ('b', 'both'):
        estimators.append(('b_selected_id_exp_logprob', np.array([float(r['score_b']) for r in rows])))
    all_policy: list[dict] = []
    all_detail: list[dict] = []
    all_summary: list[dict] = []
    for (name, conf) in estimators:
        (pol, det, summ) = run_estimator(name, conf, y, user_ids, cal_idx, test_idx)
        all_policy.extend(pol)
        all_detail.extend(det)
        all_summary.append(summ)
        print(f"{name}: test ECE raw={summ['test_ece_raw']:.4f} → iso={summ['test_ece_isotonic']:.4f}; naive P@0.9 accept={pol[0]['n_accepted']} prec={pol[0]['precision']} recall={pol[0]['recall']}; iso@0.9 accept={pol[1]['n_accepted']} prec={pol[1]['precision']} recall={pol[1]['recall']}")
    policy_path = out_dir / 'table_isotonic_threshold_policies.csv'
    with policy_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_policy[0].keys()))
        w.writeheader()
        w.writerows(all_policy)
    detail_path = out_dir / 'isotonic_test_scores.csv'
    with detail_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_detail[0].keys()))
        w.writeheader()
        w.writerows(all_detail)
    summary_path = out_dir / 'table_isotonic_summary.csv'
    flat = []
    for s in all_summary:
        flat.append({'estimator': s['estimator'], 'n_cal': s['n_cal'], 'n_test': s['n_test'], 'cal_n_correct': s['cal_n_correct'], 'test_n_correct': s['test_n_correct'], 'thr_raw_for_p90': fmt(s['thr_raw_for_p90']), 'thr_isotonic_for_p90': fmt(s['thr_isotonic_for_p90']), 'test_ece_raw': fmt(s['test_ece_raw']), 'test_ece_isotonic': fmt(s['test_ece_isotonic']), 'test_brier_raw': fmt(s['test_brier_raw']), 'test_brier_isotonic': fmt(s['test_brier_isotonic']), 'test_r90_raw': fmt(s['test_r90_raw']), 'test_r90_isotonic': fmt(s['test_r90_isotonic'])})
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    lines = ['# P3 isotonic calibration — regression_50 half-split validation', '', f'Split seed={args.seed} (P3 cal/test; **not** fixture seed 2026). n_cal={len(cal_idx)}, n_test={len(test_idx)}.', '', 'Logic check only — numbers are noisy at n=25; full-pool numbers later.', '', '## Blind τ=0.9 vs isotonic-then-0.9 (test half)', '', '| Estimator | Policy | Accept | Precision | Recall |', '|-----------|--------|--------|-----------|--------|']
    for r in all_policy:
        if r['policy'] in ('naive_raw_0.9', 'isotonic_then_0.9'):
            lines.append(f"| {r['estimator']} | {r['policy']} | {r['n_accepted']} | {r['precision']} | {r['recall']} |")
    lines.extend(['', '## Files', '', '- `table_isotonic_threshold_policies.csv`', '- `table_isotonic_summary.csv`', '- `isotonic_test_scores.csv`', ''])
    md_path = out_dir / 'DELIVERABLE_isotonic.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    meta = {'task': 'p3_isotonic_fixture_validation', 'scores_csv': str(scores_path), 'seed': args.seed, 'n_cal': int(len(cal_idx)), 'n_test': int(len(test_idx)), 'cal_user_ids': [user_ids[i] for i in cal_idx], 'test_user_ids': [user_ids[i] for i in test_idx], 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'note': 'Half-split validates code path. Do not cite fixture isotonic precision/recall as paper numbers; re-run on full pool.'}
    (out_dir / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote → {out_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
