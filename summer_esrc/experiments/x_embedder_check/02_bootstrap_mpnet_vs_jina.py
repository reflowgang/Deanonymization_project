from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esrc.paths import summer_root
N_BOOT = 10000
SEED = 2026
K = 15
DEFAULT_P1 = summer_root() / 'results' / 'runs' / 'p1_full_pool_overnight_20260805'
DEFAULT_X = summer_root() / 'results' / 'x_embedder_check'

def load_hit_at_k(path: Path, *, k: int=K) -> dict[str, bool]:
    by: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by[row['query_user_id']].append((int(row['rank']), row['candidate_user_id']))
    out: dict[str, bool] = {}
    for (uid, rows) in by.items():
        rows.sort(key=lambda x: x[0])
        cands = [cid for (_, cid) in rows[:k]]
        out[uid] = uid in cands
    return out

def load_rank1_hit(path: Path) -> dict[str, bool]:
    return load_hit_at_k(path, k=1)

def bootstrap_rate(y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(y.mean())
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    means = y[idx].mean(axis=1)
    return (point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

def bootstrap_paired_delta(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))

def mcnemar_pvalue(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int, float]:
    a = a.astype(bool)
    b = b.astype(bool)
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
    return (n11, n10, n01, n00, min(1.0, 2.0 * cdf))

def count_rows_per_query(path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            counts[row['query_user_id']] += 1
    return dict(counts)

def unique_candidates(path: Path) -> set[str]:
    cands: set[str] = set()
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cands.add(row['candidate_user_id'])
    return cands

def load_gallery_user_ids(pool: str, *, p1_dir: Path, x_dir: Path) -> tuple[set[str], set[str]]:
    if pool == 'pool_en':
        from esrc.paths import bsp_candidate_embeddings_index_csv
        idx_path = bsp_candidate_embeddings_index_csv()
    else:
        mpnet_idx = p1_dir / 'hn' / 'cache' / 'hn_candidate_embeddings_index.csv'
        jina_idx = x_dir / 'hn' / 'candidate_embeddings_index.csv'
        mpnet_ids: set[str] = set()
        jina_ids: set[str] = set()
        if mpnet_idx.exists():
            with mpnet_idx.open(newline='', encoding='utf-8') as f:
                mpnet_ids = {row['user_id'] for row in csv.DictReader(f)}
        if jina_idx.exists():
            with jina_idx.open(newline='', encoding='utf-8') as f:
                jina_ids = {row['user_id'] for row in csv.DictReader(f)}
        return (mpnet_ids, jina_ids)
    ids: set[str] = set()
    with idx_path.open(newline='', encoding='utf-8') as f:
        ids = {row['user_id'] for row in csv.DictReader(f)}
    jina_idx = x_dir / pool / 'candidate_embeddings_index.csv'
    jina_ids: set[str] = set()
    if jina_idx.exists():
        with jina_idx.open(newline='', encoding='utf-8') as f:
            jina_ids = {row['user_id'] for row in csv.DictReader(f)}
    return (ids, jina_ids)

def sanity_checks(pool: str, *, p1_dir: Path, x_dir: Path) -> dict:
    mpnet_path = p1_dir / pool / 'search_top15.csv'
    jina_path = x_dir / pool / 'search_top15_jina.csv'
    issues: list[str] = []
    mpnet_hit = load_hit_at_k(mpnet_path)
    jina_hit = load_hit_at_k(jina_path)
    mpnet_ids = set(mpnet_hit)
    jina_ids = set(jina_hit)
    if mpnet_ids != jina_ids:
        only_mp = sorted(mpnet_ids - jina_ids)[:5]
        only_j = sorted(jina_ids - mpnet_ids)[:5]
        issues.append(f'query set mismatch: mpnet={len(mpnet_ids)} jina={len(jina_ids)} only_mpnet_sample={only_mp} only_jina_sample={only_j}')
    for (label, path) in (('mpnet', mpnet_path), ('jina', jina_path)):
        counts = count_rows_per_query(path)
        bad = {u: n for (u, n) in counts.items() if n != K}
        if bad:
            issues.append(f'{label}: {len(bad)} queries with rows != {K}')
    mpnet_cands_in_results = unique_candidates(mpnet_path)
    jina_cands_in_results = unique_candidates(jina_path)
    (gallery_mpnet, gallery_jina) = load_gallery_user_ids(pool, p1_dir=p1_dir, x_dir=x_dir)
    if gallery_mpnet and gallery_jina and (gallery_mpnet != gallery_jina):
        issues.append(f'gallery index mismatch: mpnet={len(gallery_mpnet)} jina={len(gallery_jina)} symmetric_diff={len(gallery_mpnet ^ gallery_jina)}')
    elif not gallery_mpnet or not gallery_jina:
        issues.append('missing gallery index CSV for mpnet or jina cache')
    metrics_path = x_dir / pool / 'metrics.json'
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        recomputed = sum(mpnet_hit.values()) / len(mpnet_hit)
        stored = metrics['mpnet_baseline']['hit_at_15']
        if abs(recomputed - stored) > 0.0001:
            issues.append(f'mpnet Hit@15 recompute {recomputed:.6f} != metrics.json {stored:.6f}')
    paired = sorted(mpnet_ids & jina_ids)
    mp_y = np.array([1.0 if mpnet_hit[u] else 0.0 for u in paired])
    j_y = np.array([1.0 if jina_hit[u] else 0.0 for u in paired])
    both = mp_y.astype(bool) & j_y.astype(bool)
    mp_only = mp_y.astype(bool) & ~j_y.astype(bool)
    j_only = ~mp_y.astype(bool) & j_y.astype(bool)
    return {'pool': pool, 'n_paired': len(paired), 'gallery_size_mpnet': len(gallery_mpnet), 'gallery_size_jina': len(gallery_jina), 'unique_candidates_in_top15_mpnet': len(mpnet_cands_in_results), 'unique_candidates_in_top15_jina': len(jina_cands_in_results), 'mpnet_hit_at_15_recomputed': round(float(mp_y.mean()), 4), 'jina_hit_at_15_recomputed': round(float(j_y.mean()), 4), 'discordant_mpnet_only': int(mp_only.sum()), 'discordant_jina_only': int(j_only.sum()), 'both_hit': int(both.sum()), 'neither_hit': int((~mp_y.astype(bool) & ~j_y.astype(bool)).sum()), 'issues': issues, 'ok': len(issues) == 0}

def main() -> None:
    p = argparse.ArgumentParser(description='Bootstrap mpnet vs jina Hit@15')
    p.add_argument('--p1-run-dir', default=str(DEFAULT_P1))
    p.add_argument('--x-dir', default=str(DEFAULT_X))
    p.add_argument('--out-dir', default=str(DEFAULT_X))
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    p1_dir = Path(args.p1_run_dir)
    x_dir = Path(args.x_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sanity_rows: list[dict] = []
    boot_rows: list[dict] = []
    for pool in ('pool_en', 'hn'):
        check = sanity_checks(pool, p1_dir=p1_dir, x_dir=x_dir)
        sanity_rows.append(check)
        if check['issues']:
            print(f'[{pool}] SANITY ISSUES:')
            for issue in check['issues']:
                print(f'  - {issue}')
        else:
            print(f"[{pool}] sanity OK: n={check['n_paired']} gallery={check['gallery_size_mpnet']}/{check['gallery_size_jina']}")
        mpnet_path = p1_dir / pool / 'search_top15.csv'
        jina_path = x_dir / pool / 'search_top15_jina.csv'
        mpnet_hit = load_hit_at_k(mpnet_path)
        jina_hit = load_hit_at_k(jina_path)
        ids = sorted(set(mpnet_hit) & set(jina_hit))
        mp_y = np.array([1.0 if mpnet_hit[u] else 0.0 for u in ids])
        j_y = np.array([1.0 if jina_hit[u] else 0.0 for u in ids])
        (mp_p, mp_lo, mp_hi) = bootstrap_rate(mp_y, n_boot=args.n_boot, seed=args.seed)
        (j_p, j_lo, j_hi) = bootstrap_rate(j_y, n_boot=args.n_boot, seed=args.seed + 1)
        (d_p, d_lo, d_hi) = bootstrap_paired_delta(j_y, mp_y, n_boot=args.n_boot, seed=args.seed + 2)
        (n11, n10, n01, n00, p_mc) = mcnemar_pvalue(j_y, mp_y)
        sig = not d_lo <= 0.0 <= d_hi
        boot_rows.append({'pool': pool, 'metric': 'hit_at_15', 'n_paired': len(ids), 'mpnet_rate': round(mp_p, 4), 'mpnet_ci_low': round(mp_lo, 4), 'mpnet_ci_high': round(mp_hi, 4), 'mpnet_pct': f'{100 * mp_p:.1f}% [{100 * mp_lo:.1f}-{100 * mp_hi:.1f}]', 'jina_rate': round(j_p, 4), 'jina_ci_low': round(j_lo, 4), 'jina_ci_high': round(j_hi, 4), 'jina_pct': f'{100 * j_p:.1f}% [{100 * j_lo:.1f}-{100 * j_hi:.1f}]', 'delta_jina_minus_mpnet': round(d_p, 4), 'delta_ci_low': round(d_lo, 4), 'delta_ci_high': round(d_hi, 4), 'delta_pp': round(100 * d_p, 2), 'delta_ci_pp': f'[{100 * d_lo:+.1f}, {100 * d_hi:+.1f}]', 'delta_ci_excludes_zero': sig, 'mcnemar_n11_both': n11, 'mcnemar_n10_jina_only': n10, 'mcnemar_n01_mpnet_only': n01, 'mcnemar_n00_neither': n00, 'mcnemar_p': round(p_mc, 6), 'n_boot': args.n_boot, 'seed': args.seed})
        print(f'[{pool}] Hit@15 mpnet={100 * mp_p:.1f}% [{100 * mp_lo:.1f}-{100 * mp_hi:.1f}] jina={100 * j_p:.1f}% [{100 * j_lo:.1f}-{100 * j_hi:.1f}] Δ={100 * d_p:+.1f}pp {100 * d_lo:+.1f},{100 * d_hi:+.1f} excludes0={sig} McNemar p={p_mc:.2e} paired={len(ids)}')
    sanity_path = out_dir / 'sanity_checks.json'
    sanity_path.write_text(json.dumps(sanity_rows, indent=2) + '\n', encoding='utf-8')
    boot_path = out_dir / 'bootstrap_mpnet_vs_jina_hit15.csv'
    with boot_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        w.writerows(boot_rows)
    print('wrote', sanity_path)
    print('wrote', boot_path)
if __name__ == '__main__':
    main()
