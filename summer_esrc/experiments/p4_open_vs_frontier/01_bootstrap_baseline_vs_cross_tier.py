from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.paths import summer_root
N_BOOT = 10000
SEED = 2026

def last_ok_reason(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            by[o['query_user_id']] = o
    return {u: o for (u, o) in by.items() if o.get('status') == 'ok'}

def load_search_hit(path: Path) -> dict[str, bool]:
    by: dict[str, list[str]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by[row['query_user_id']].append(row['candidate_user_id'])
    return {q: q in cands for (q, cands) in by.items()}

def load_local(pool: str, run_dir: Path) -> tuple[dict[str, bool], dict[str, bool]]:
    preds = last_ok_reason(run_dir / pool / 'reason_predictions.jsonl')
    top1 = {u: bool(o.get('correct')) for (u, o) in preds.items()}
    hit_from_pred = {u: bool(o.get('true_in_top15')) for (u, o) in preds.items() if 'true_in_top15' in o}
    if len(hit_from_pred) == len(preds):
        hit = hit_from_pred
    else:
        hit = load_search_hit(run_dir / pool / 'search_top15.csv')
        hit = {u: hit.get(u, False) for u in preds}
    return (top1, hit)

def bootstrap_rate(y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    point = float(y.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = y[idx].mean(axis=1)
    return (point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

def bootstrap_paired_delta(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    n = len(a)
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))

def mcnemar_pvalue(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int, float]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    n_disc = n10 + n01
    if n_disc == 0:
        return (n11, n10, n01, n00, 1.0)
    from math import comb
    k = min(n10, n01)
    cdf = sum((comb(n_disc, i) for i in range(0, k + 1))) / 2 ** n_disc
    p = min(1.0, 2.0 * cdf)
    return (n11, n10, n01, n00, p)

def aligned_vectors(a: dict[str, bool], b: dict[str, bool]) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(a) & set(b))
    va = np.array([1.0 if a[u] else 0.0 for u in ids], dtype=float)
    vb = np.array([1.0 if b[u] else 0.0 for u in ids], dtype=float)
    return (ids, va, vb)

def pct(x: float) -> str:
    if x != x:
        return ''
    return f'{100.0 * x:.1f}'

def main() -> None:
    p = argparse.ArgumentParser(description='P4 C2 bootstrap: baseline vs cross-tier')
    p.add_argument('--baseline-dir', default=str(summer_root() / 'results' / 'runs' / 'p1_full_pool_overnight_20260805'))
    p.add_argument('--cross-tier-dir', default=str(summer_root() / 'results' / 'runs' / 'p4_cross_tier_full_pool'))
    p.add_argument('--out-dir', default=str(summer_root() / 'results' / 'p4_open_vs_frontier'))
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    base_dir = Path(args.baseline_dir)
    xt_dir = Path(args.cross_tier_dir)
    out_dir = Path(args.out_dir)
    tables = out_dir / 'tables_c2'
    tables.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    mcnemar_rows: list[dict] = []
    for pool in ('pool_en', 'hn'):
        (base_top1, base_hit) = load_local(pool, base_dir)
        (xt_top1, xt_hit) = load_local(pool, xt_dir)
        for (metric, base_d, xt_d) in (('top1', base_top1, xt_top1), ('hit_at_15', base_hit, xt_hit)):
            (ids, xt_y, base_y) = aligned_vectors(xt_d, base_d)
            base_ids = sorted(base_d)
            xt_ids = sorted(xt_d)
            base_full = np.array([1.0 if base_d[u] else 0.0 for u in base_ids])
            xt_full = np.array([1.0 if xt_d[u] else 0.0 for u in xt_ids])
            (base_p, base_lo, base_hi) = bootstrap_rate(base_full, n_boot=args.n_boot, seed=args.seed)
            (xt_p, xt_lo, xt_hi) = bootstrap_rate(xt_full, n_boot=args.n_boot, seed=args.seed)
            (d_p, d_lo, d_hi) = bootstrap_paired_delta(xt_y, base_y, n_boot=args.n_boot, seed=args.seed)
            (n11, n10, n01, n00, p_mc) = mcnemar_pvalue(xt_y.astype(bool), base_y.astype(bool))
            significant = not d_lo <= 0.0 <= d_hi
            rows.append({'pool': pool, 'metric': metric, 'n_baseline': len(base_ids), 'n_cross_tier': len(xt_ids), 'n_paired': len(ids), 'baseline_rate': round(base_p, 4), 'baseline_ci_low': round(base_lo, 4), 'baseline_ci_high': round(base_hi, 4), 'baseline_pct': pct(base_p), 'cross_tier_rate': round(xt_p, 4), 'cross_tier_ci_low': round(xt_lo, 4), 'cross_tier_ci_high': round(xt_hi, 4), 'cross_tier_pct': pct(xt_p), 'delta_cross_minus_base': round(d_p, 4), 'delta_ci_low': round(d_lo, 4), 'delta_ci_high': round(d_hi, 4), 'delta_pct_pts': round(100.0 * d_p, 2), 'delta_ci_excludes_zero': significant, 'mcnemar_n11': n11, 'mcnemar_n10_cross_only': n10, 'mcnemar_n01_base_only': n01, 'mcnemar_n00': n00, 'mcnemar_p': round(p_mc, 6), 'n_boot': args.n_boot, 'seed': args.seed})
            mcnemar_rows.append({'pool': pool, 'metric': metric, 'n_paired': len(ids), 'n11_both': n11, 'n10_cross_only': n10, 'n01_base_only': n01, 'n00_neither': n00, 'p_value': round(p_mc, 6), 'significant_0_05': p_mc < 0.05})
            print(f'{pool} {metric}: base={pct(base_p)}% [{pct(base_lo)}-{pct(base_hi)}] cross={pct(xt_p)}% [{pct(xt_lo)}-{pct(xt_hi)}] Δ={100 * d_p:+.1f}pp [{100 * d_lo:+.1f},{100 * d_hi:+.1f}] excludes0={significant} McNemar p={p_mc:.4g} paired={len(ids)}')
    boot_path = tables / 'baseline_vs_cross_tier_bootstrap_ci.csv'
    with boot_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    mc_path = tables / 'baseline_vs_cross_tier_mcnemar.csv'
    with mc_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
        w.writeheader()
        w.writerows(mcnemar_rows)
    compact = []
    for r in rows:
        compact.append({'pool': r['pool'], 'metric': r['metric'], 'n_paired': r['n_paired'], 'baseline': f"{r['baseline_pct']}% [{pct(r['baseline_ci_low'])}-{pct(r['baseline_ci_high'])}]", 'cross_tier': f"{r['cross_tier_pct']}% [{pct(r['cross_tier_ci_low'])}-{pct(r['cross_tier_ci_high'])}]", 'delta_pp': f"{r['delta_pct_pts']:+.1f} [{100 * r['delta_ci_low']:+.1f}, {100 * r['delta_ci_high']:+.1f}]", 'mcnemar_p': r['mcnemar_p'], 'delta_ci_excludes_zero': r['delta_ci_excludes_zero']})
    compact_path = tables / 'C2_comparison_table.csv'
    with compact_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(compact[0].keys()))
        w.writeheader()
        w.writerows(compact)
    print('wrote', boot_path)
    print('wrote', mc_path)
    print('wrote', compact_path)
if __name__ == '__main__':
    main()
