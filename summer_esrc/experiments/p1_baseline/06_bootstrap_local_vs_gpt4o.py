#!/usr/bin/env python3
"""P1 deliverable — bootstrap CI: local open-weight vs archived gpt-4o (T8).

Claims from brief:
  C1: Local stack ≈ gpt-4o on Reddit (pool_en) top-1 / Hit@15
  C2: Local stack transfers to HN at gpt-4o level

Uses finalized local preds (500/500 pool_en, 499/499 HN) and archived
gpt-4o reason + FAISS tables. Bootstrap n=10_000, seed=2026.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BSP = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from esrc.paths import summer_root  # noqa: E402

N_BOOT = 10_000
SEED = 2026


def last_ok_reason(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            by[o["query_user_id"]] = o
    return {u: o for u, o in by.items() if o.get("status") == "ok"}


def load_search_hit(path: Path, *, level_key: Optional[str] = None, level: Optional[str] = None) -> dict[str, bool]:
    by: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if level_key and level and row.get(level_key) != level:
                continue
            by[row["query_user_id"]].append(row["candidate_user_id"])
    return {q: q in cands for q, cands in by.items()}


def load_gpt4o_pool_en_t8() -> tuple[dict[str, bool], dict[str, bool]]:
    pred_path = BSP / "results" / "tables" / "pool_en_reason_predictions.csv"
    top1: dict[str, bool] = {}
    with pred_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["T_level"] != "T8" or row.get("status") != "ok":
                continue
            uid = row["query_user_id"]
            top1[uid] = uid == row["predicted_candidate_user_id"]
    hit = load_search_hit(
        BSP / "results" / "tables" / "pool_en_faiss_top15.csv",
        level_key="T_level",
        level="T8",
    )
    return top1, hit


def load_gpt4o_hn_t8() -> tuple[dict[str, bool], dict[str, bool]]:
    pred_path = BSP / "results" / "tables" / "hn_reason_predictions.csv"
    top1: dict[str, bool] = {}
    with pred_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["level"] != "T8" or row.get("status") != "success":
                continue
            uid = row["query_user_id"]
            top1[uid] = str(row.get("is_correct", "")).strip() in ("1", "true", "True")
    hit = load_search_hit(
        BSP / "results" / "tables" / "hn_faiss_top15.csv",
        level_key="level",
        level="T8",
    )
    # Prefer is_true_match column if present for consistency check
    return top1, hit


def load_local(pool: str, run_dir: Path) -> tuple[dict[str, bool], dict[str, bool]]:
    preds = last_ok_reason(run_dir / pool / "reason_predictions.jsonl")
    top1 = {u: bool(o.get("correct")) for u, o in preds.items()}
    # Prefer true_in_top15 from reason row when present; else search CSV
    hit_from_pred = {
        u: bool(o.get("true_in_top15"))
        for u, o in preds.items()
        if "true_in_top15" in o
    }
    if len(hit_from_pred) == len(preds):
        hit = hit_from_pred
    else:
        hit = load_search_hit(run_dir / pool / "search_top15.csv")
    return top1, hit


def bootstrap_rate(y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(y.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = y[idx].mean(axis=1)
    return point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_paired_delta(
    a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Delta = local - gpt4o on paired users."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    n = len(a)
    rng = np.random.default_rng(seed)
    point = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def mcnemar_pvalue(local: np.ndarray, ref: np.ndarray) -> tuple[int, int, int, int, float]:
    """Exact binomial McNemar on discordant pairs; returns n11,n10,n01,n00,p."""
    local = np.asarray(local, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    n11 = int((local & ref).sum())
    n10 = int((local & ~ref).sum())
    n01 = int((~local & ref).sum())
    n00 = int((~local & ~ref).sum())
    n_disc = n10 + n01
    if n_disc == 0:
        return n11, n10, n01, n00, 1.0
    # two-sided exact binomial under p=0.5
    k = min(n10, n01)
    # P(|X - n/2| >= |n10 - n/2|) = 2 * cdf(k) but careful when equal
    from math import comb

    cdf = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2**n_disc)
    p = min(1.0, 2.0 * cdf)
    return n11, n10, n01, n00, p


def aligned_vectors(
    local: dict[str, bool], ref: dict[str, bool]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(local) & set(ref))
    a = np.array([1.0 if local[u] else 0.0 for u in ids], dtype=float)
    b = np.array([1.0 if ref[u] else 0.0 for u in ids], dtype=float)
    return ids, a, b


def pct(x: float) -> str:
    if x != x:
        return ""
    return f"{100.0 * x:.1f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        default=str(
            summer_root() / "results" / "runs" / "p1_full_pool_overnight_20260805"
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(summer_root() / "results" / "p1_baseline"),
    )
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    preds_dir = out_dir / "predictions"
    tables.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    gpt_pe_top1, gpt_pe_hit = load_gpt4o_pool_en_t8()
    gpt_hn_top1, gpt_hn_hit = load_gpt4o_hn_t8()
    loc_pe_top1, loc_pe_hit = load_local("pool_en", run_dir)
    loc_hn_top1, loc_hn_hit = load_local("hn", run_dir)

    # Write clean prediction deliverables (last ok row)
    for pool in ("pool_en", "hn"):
        src = last_ok_reason(run_dir / pool / "reason_predictions.jsonl")
        out_jsonl = preds_dir / f"local_{pool}_T8_reason_predictions.jsonl"
        out_csv = preds_dir / f"local_{pool}_T8_reason_predictions.csv"
        fields = [
            "pool",
            "T_level",
            "query_user_id",
            "selected_candidate_user_id",
            "selected_candidate_number",
            "correct",
            "true_in_top15",
            "model",
            "status",
        ]
        with out_jsonl.open("w", encoding="utf-8") as fj, out_csv.open(
            "w", newline="", encoding="utf-8"
        ) as fc:
            w = csv.DictWriter(fc, fieldnames=fields)
            w.writeheader()
            for uid in sorted(src):
                o = src[uid]
                row = {k: o.get(k, "") for k in fields}
                row["pool"] = pool
                row["T_level"] = o.get("T_level", "T8")
                fj.write(json.dumps(o) + "\n")
                w.writerow(row)
        print(f"wrote {out_jsonl} n={len(src)}")

    rows: list[dict] = []
    mcnemar_rows: list[dict] = []

    specs = [
        ("pool_en", "top1", "C1", loc_pe_top1, gpt_pe_top1),
        ("pool_en", "hit_at_15", "C1", loc_pe_hit, gpt_pe_hit),
        ("hn", "top1", "C2", loc_hn_top1, gpt_hn_top1),
        ("hn", "hit_at_15", "C2", loc_hn_hit, gpt_hn_hit),
    ]

    for pool, metric, claim, local_d, ref_d in specs:
        ids, a, b = aligned_vectors(local_d, ref_d)
        # Also report full local / full ref denominators (may differ on HN)
        loc_ids = sorted(local_d)
        ref_ids = sorted(ref_d)
        loc_y = np.array([1.0 if local_d[u] else 0.0 for u in loc_ids])
        ref_y = np.array([1.0 if ref_d[u] else 0.0 for u in ref_ids])

        loc_p, loc_lo, loc_hi = bootstrap_rate(loc_y, n_boot=args.n_boot, seed=args.seed)
        ref_p, ref_lo, ref_hi = bootstrap_rate(ref_y, n_boot=args.n_boot, seed=args.seed)
        d_p, d_lo, d_hi = bootstrap_paired_delta(a, b, n_boot=args.n_boot, seed=args.seed)
        n11, n10, n01, n00, p_mc = mcnemar_pvalue(a.astype(bool), b.astype(bool))

        # Decision heuristic: CIs on delta include 0 → no significant gap
        significant_gap = not (d_lo <= 0.0 <= d_hi)
        supports_claim = (not significant_gap) if claim == "C1" else (not significant_gap and abs(d_p) < 0.05)
        # C1: support if local≈ref (delta CI includes 0)
        # C2: support if local≈ref on HN (same); clearly reject if large negative delta

        rows.append(
            {
                "claim": claim,
                "pool": pool,
                "metric": metric,
                "n_local": len(loc_ids),
                "n_gpt4o": len(ref_ids),
                "n_paired": len(ids),
                "local_rate": round(loc_p, 4),
                "local_ci_low": round(loc_lo, 4),
                "local_ci_high": round(loc_hi, 4),
                "local_pct": pct(loc_p),
                "gpt4o_rate": round(ref_p, 4),
                "gpt4o_ci_low": round(ref_lo, 4),
                "gpt4o_ci_high": round(ref_hi, 4),
                "gpt4o_pct": pct(ref_p),
                "delta_local_minus_gpt4o": round(d_p, 4),
                "delta_ci_low": round(d_lo, 4),
                "delta_ci_high": round(d_hi, 4),
                "delta_pct_pts": round(100.0 * d_p, 2),
                "delta_ci_excludes_zero": significant_gap,
                "mcnemar_n11": n11,
                "mcnemar_n10_local_only": n10,
                "mcnemar_n01_gpt4o_only": n01,
                "mcnemar_n00": n00,
                "mcnemar_p": round(p_mc, 6),
                "n_boot": args.n_boot,
                "seed": args.seed,
            }
        )
        mcnemar_rows.append(
            {
                "claim": claim,
                "pool": pool,
                "metric": metric,
                "n_paired": len(ids),
                "n11_both": n11,
                "n10_local_only": n10,
                "n01_gpt4o_only": n01,
                "n00_neither": n00,
                "p_value": round(p_mc, 6),
                "significant_0_05": p_mc < 0.05,
            }
        )
        print(
            f"{claim} {pool} {metric}: local={pct(loc_p)}% [{pct(loc_lo)}-{pct(loc_hi)}] "
            f"gpt4o={pct(ref_p)}% [{pct(ref_lo)}-{pct(ref_hi)}] "
            f"Δ={100*d_p:+.1f}pp [{100*d_lo:+.1f},{100*d_hi:+.1f}] "
            f"McNemar p={p_mc:.4g} paired={len(ids)}"
        )

    cmp_path = tables / "local_vs_gpt4o_bootstrap_ci.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mc_path = tables / "local_vs_gpt4o_mcnemar.csv"
    with mc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0].keys()))
        w.writeheader()
        w.writerows(mcnemar_rows)

    # Compact hand-in comparison table (percentages)
    compact = []
    for r in rows:
        compact.append(
            {
                "claim": r["claim"],
                "pool": r["pool"],
                "metric": r["metric"],
                "n_paired": r["n_paired"],
                "local": f"{r['local_pct']}% [{pct(r['local_ci_low'])}-{pct(r['local_ci_high'])}]",
                "gpt4o": f"{r['gpt4o_pct']}% [{pct(r['gpt4o_ci_low'])}-{pct(r['gpt4o_ci_high'])}]",
                "delta_pp": f"{r['delta_pct_pts']:+.1f} [{100*r['delta_ci_low']:+.1f}, {100*r['delta_ci_high']:+.1f}]",
                "mcnemar_p": r["mcnemar_p"],
                "delta_ci_excludes_zero": r["delta_ci_excludes_zero"],
            }
        )
    compact_path = tables / "P1_comparison_table.csv"
    with compact_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(compact[0].keys()))
        w.writeheader()
        w.writerows(compact)

    print("wrote", cmp_path)
    print("wrote", mc_path)
    print("wrote", compact_path)


if __name__ == "__main__":
    main()
