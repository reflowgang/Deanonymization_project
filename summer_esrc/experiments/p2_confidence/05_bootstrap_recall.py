#!/usr/bin/env python3
"""P2.12 — Bootstrap CIs for Recall@90/99%P on estimator_scores.csv (fixture-scale)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap Recall@P CIs for P2 estimators")
    p.add_argument("--scores-csv", type=str, required=True)
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def as_bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes"}


def recall_at_precision(conf: np.ndarray, y: np.ndarray, target: float) -> float | None:
    if y.sum() == 0:
        return None
    precision, recall, _ = precision_recall_curve(y, conf)
    ok = precision[:-1] >= target
    if not np.any(ok):
        if float(precision[-1]) >= target:
            return float(recall[-1])
        return None
    return float(np.max(recall[:-1][ok]))


def bootstrap(conf: np.ndarray, y: np.ndarray, target: float, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    point = recall_at_precision(conf, y, target)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, cb = y[idx], conf[idx]
        if yb.sum() == 0:
            vals.append(np.nan)
            continue
        v = recall_at_precision(cb, yb, target)
        vals.append(np.nan if v is None else v)
    arr = np.asarray(vals, dtype=float)
    none_as_zero = np.nan_to_num(arr, nan=0.0)
    defined = arr[~np.isnan(arr)]
    return {
        "point_raw": point,
        "ci_none_as_zero": (
            float(np.quantile(none_as_zero, 0.025)),
            float(np.quantile(none_as_zero, 0.975)),
        ),
        "mean_none_as_zero": float(none_as_zero.mean()),
        "ci_defined_only": (
            (float(np.quantile(defined, 0.025)), float(np.quantile(defined, 0.975)))
            if len(defined)
            else (None, None)
        ),
        "n_defined": int(len(defined)),
        "n_undefined": int(np.isnan(arr).sum()),
    }


def main() -> int:
    args = parse_args()
    path = Path(args.scores_csv)
    rows = [r for r in csv.DictReader(path.open(newline="", encoding="utf-8")) if r.get("status") == "ok"]
    y = np.array([as_bool(r["correct"]) for r in rows], dtype=int)
    score_map = {
        "a_verbalized": np.array([float(r["score_a"]) for r in rows]),
        "b_selected_id_exp_logprob": np.array([float(r["score_b"]) for r in rows]),
        "c_softmax_on_selected": np.array([float(r["score_c"]) for r in rows]),
    }

    out_rows = []
    for name, conf in score_map.items():
        for target, metric in [(0.90, "recall_at_90_precision"), (0.99, "recall_at_99_precision")]:
            b = bootstrap(conf, y, target, args.n_boot, args.seed)
            lo, hi = b["ci_none_as_zero"]
            dlo, dhi = b["ci_defined_only"]
            out_rows.append(
                {
                    "estimator": name,
                    "metric": metric,
                    "n": len(rows),
                    "n_correct": int(y.sum()),
                    "point_estimate": "" if b["point_raw"] is None else f"{b['point_raw']:.6f}",
                    "boot_mean_none_as_zero": f"{b['mean_none_as_zero']:.6f}",
                    "ci95_low_none_as_zero": f"{lo:.6f}",
                    "ci95_high_none_as_zero": f"{hi:.6f}",
                    "ci95_low_defined_only": "" if dlo is None else f"{dlo:.6f}",
                    "ci95_high_defined_only": "" if dhi is None else f"{dhi:.6f}",
                    "n_boot_defined": b["n_defined"],
                    "n_boot_undefined": b["n_undefined"],
                    "n_boot": args.n_boot,
                    "seed": args.seed,
                }
            )
            print(
                f"{name} {metric}: point={b['point_raw']} "
                f"95%CI(none→0)=[{lo:.3f}, {hi:.3f}] undefined={b['n_undefined']}/{args.n_boot}"
            )

    out = Path(args.out_csv) if args.out_csv else path.with_name("bootstrap_recall_at_precision.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
