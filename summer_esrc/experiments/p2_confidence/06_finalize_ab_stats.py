from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.metrics_calibration import bonferroni_alpha, bootstrap_metric_ci, brier_score, evaluate_scores, expected_calibration_error, mcnemar_threshold_match, recall_at_precision, reliability_table
from esrc.paths import summer_root
DEFAULT_SCORES = summer_root() / 'results/runs/p2_regression_50_T8/estimator_scores.csv'
OUT_DIR = summer_root() / 'results/p2_confidence/tables'
N_BOOT = 10000
SEED = 2026
ALPHA = 0.05
N_MCNEMAR_TESTS = 2

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Finalize P2 (a)/(b) stats tables')
    p.add_argument('--scores-csv', type=str, default=str(DEFAULT_SCORES))
    p.add_argument('--out-dir', type=str, default=str(OUT_DIR))
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()

def as_bool(v: str) -> bool:
    return str(v).lower() in {'1', 'true', 'yes'}

def fmt(x: float | None, digits: int=4) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and x != x:
        return ''
    return f'{x:.{digits}f}'

def main() -> int:
    args = parse_args()
    scores_path = Path(args.scores_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(scores_path.open(newline='', encoding='utf-8')) if r.get('status') == 'ok']
    if not rows:
        raise SystemExit(f'No ok rows in {scores_path}')
    y = [as_bool(r['correct']) for r in rows]
    score_a = [float(r['score_a']) for r in rows]
    score_b = [float(r['score_b']) for r in rows]
    user_ids = [r['query_user_id'] for r in rows]
    summary_rows = []
    for (label, conf) in [('a_verbalized', score_a), ('b_selected_id_exp_logprob', score_b)]:
        m = evaluate_scores(conf, y)
        summary_rows.append({'estimator': label, 'n': m.n, 'n_correct': m.n_correct, 'top1_accuracy': fmt(m.top1_accuracy, 4), 'ece': fmt(m.ece, 4), 'brier': fmt(m.brier, 4), 'average_precision': fmt(m.average_precision, 4), 'recall_at_90_precision': fmt(m.recall_at_90_precision, 4), 'recall_at_99_precision': fmt(m.recall_at_99_precision, 4)})
    summary_path = out_dir / 'table_ab_metrics.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    boot_rows = []

    def _ap(c, yy):
        yy = np.asarray(yy)
        if yy.sum() == 0:
            return None
        if yy.sum() == len(yy):
            return 1.0
        return float(average_precision_score(yy, c))
    metrics_to_boot = [('recall_at_90_precision', lambda c, yy: recall_at_precision(c, yy, 0.9)), ('recall_at_99_precision', lambda c, yy: recall_at_precision(c, yy, 0.99)), ('ece', lambda c, yy: float(expected_calibration_error(c, yy))), ('brier', lambda c, yy: float(brier_score(c, yy))), ('average_precision', _ap)]
    for (label, conf) in [('a_verbalized', score_a), ('b_selected_id_exp_logprob', score_b)]:
        for (metric_name, fn) in metrics_to_boot:
            undef_zero = metric_name.startswith('recall_at_')
            ci = bootstrap_metric_ci(conf, y, fn, n_boot=args.n_boot, seed=args.seed, undefined_as_zero=undef_zero)
            boot_rows.append({'estimator': label, 'metric': metric_name, 'n': len(rows), 'point': fmt(ci.point, 4), 'boot_mean': fmt(ci.mean, 4), 'ci95_low': fmt(ci.ci_low, 4), 'ci95_high': fmt(ci.ci_high, 4), 'n_boot': ci.n_boot, 'n_boot_undefined': ci.n_undefined, 'undefined_as': ci.undefined_as, 'seed': args.seed})
            print(f'{label} {metric_name}: {ci.point} 95%CI=[{ci.ci_low:.4f}, {ci.ci_high:.4f}]')
    boot_path = out_dir / 'table_ab_bootstrap_ci.csv'
    with boot_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        w.writerows(boot_rows)
    alpha_adj = bonferroni_alpha(ALPHA, N_MCNEMAR_TESTS)
    mcnemar_rows = []
    for tau in (0.9, 0.99):
        res = mcnemar_threshold_match(score_a, score_b, y, threshold=tau)
        mcnemar_rows.append({'comparison': 'a_vs_b', 'threshold': tau, 'n11_both_match': res.n11, 'n10_a_only': res.n10, 'n01_b_only': res.n01, 'n00_neither': res.n00, 'n_discordant': res.n_discordant, 'statistic': fmt(res.statistic, 4), 'p_value': fmt(res.p_value, 6), 'alpha': ALPHA, 'alpha_bonferroni': fmt(alpha_adj, 6), 'significant_bonferroni': str(res.p_value < alpha_adj).lower(), 'note': res.note})
        print(f'McNemar τ={tau}: n10={res.n10} n01={res.n01} stat={res.statistic:.4f} p={res.p_value:.6f} sig@α/m={res.p_value < alpha_adj}')
    mcnemar_path = out_dir / 'table_ab_mcnemar.csv'
    with mcnemar_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
        w.writeheader()
        w.writerows(mcnemar_rows)
    rel_rows = []
    for (label, conf) in [('a_verbalized', score_a), ('b_selected_id_exp_logprob', score_b)]:
        for b in reliability_table(conf, y, n_bins=10):
            rel_rows.append({'estimator': label, 'bin_lo': fmt(b.bin_lo, 2), 'bin_hi': fmt(b.bin_hi, 2), 'n': b.n, 'mean_confidence': fmt(b.mean_confidence, 4), 'accuracy': fmt(b.accuracy, 4), 'gap_abs': fmt(b.gap, 4)})
    rel_path = out_dir / 'table_ab_reliability.csv'
    with rel_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rel_rows[0].keys()))
        w.writeheader()
        w.writerows(rel_rows)

    def _ci(est: str, metric: str) -> tuple[str, str, str]:
        for r in boot_rows:
            if r['estimator'] == est and r['metric'] == metric:
                return (r['point'], r['ci95_low'], r['ci95_high'])
        return ('', '', '')
    (a90_p, a90_lo, a90_hi) = _ci('a_verbalized', 'recall_at_90_precision')
    (b90_p, b90_lo, b90_hi) = _ci('b_selected_id_exp_logprob', 'recall_at_90_precision')
    (a99_p, a99_lo, a99_hi) = _ci('a_verbalized', 'recall_at_99_precision')
    (b99_p, b99_lo, b99_hi) = _ci('b_selected_id_exp_logprob', 'recall_at_99_precision')
    ma = summary_rows[0]
    mb = summary_rows[1]
    mc09 = mcnemar_rows[0]
    md = f"# P2 estimators (a) vs (b) — regression_50 deliverable\n\nSource run: `{scores_path}`  \nn={len(rows)}, top-1 correct={sum(y)}/{len(rows)}, bootstrap n={args.n_boot}, seed={args.seed}.\n\n## Summary\n\n| Estimator | ECE | Brier | AP | R@90%P | 95% CI | R@99%P | 95% CI |\n|-----------|-----|-------|-----|--------|--------|--------|--------|\n| (a) verbalized | {ma['ece']} | {ma['brier']} | {ma['average_precision']} | {a90_p} | [{a90_lo}, {a90_hi}] | {a99_p} | [{a99_lo}, {a99_hi}] |\n| (b) selected-id exp(logprob) | {mb['ece']} | {mb['brier']} | {mb['average_precision']} | {b90_p} | [{b90_lo}, {b90_hi}] | {b99_p} | [{b99_lo}, {b99_hi}] |\n\n## McNemar (a) vs (b) — thresholded correctness classifiers\n\nPredict “correct” iff conf ≥ τ. Continuity-corrected McNemar; Bonferroni α/m with m={N_MCNEMAR_TESTS} (τ=0.9, 0.99).\n\n| τ | n10 (a only) | n01 (b only) | disc. | χ² | p | sig @ α/m={fmt(alpha_adj, 4)} |\n|---|--------------|--------------|-------|----|---|-------------------------------|\n| 0.9 | {mc09['n10_a_only']} | {mc09['n01_b_only']} | {mc09['n_discordant']} | {mc09['statistic']} | {mc09['p_value']} | {mc09['significant_bonferroni']} |\n| 0.99 | {mcnemar_rows[1]['n10_a_only']} | {mcnemar_rows[1]['n01_b_only']} | {mcnemar_rows[1]['n_discordant']} | {mcnemar_rows[1]['statistic']} | {mcnemar_rows[1]['p_value']} | {mcnemar_rows[1]['significant_bonferroni']} |\n\n**Note:** (a) and (b) share the same Reason pick; McNemar compares the *confidence-threshold classifiers*, not pick accuracy. n=50 CIs are wide — treat as fixture-scale diagnostics, not pool claims.\n\n## Files\n\n- `table_ab_metrics.csv`\n- `table_ab_bootstrap_ci.csv`\n- `table_ab_mcnemar.csv`\n- `table_ab_reliability.csv`\n"
    md_path = out_dir / 'DELIVERABLE_ab.md'
    md_path.write_text(md, encoding='utf-8')
    meta = {'task': 'p2_finalize_ab', 'scores_csv': str(scores_path), 'n': len(rows), 'n_correct': sum(y), 'n_boot': args.n_boot, 'seed': args.seed, 'bonferroni_alpha': alpha_adj, 'user_ids_sha_prefix': ''.join(user_ids)[:16], 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'outputs': [str(summary_path), str(boot_path), str(mcnemar_path), str(rel_path), str(md_path)]}
    (out_dir / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote deliverables → {out_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
