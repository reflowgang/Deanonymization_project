#!/usr/bin/env python3
"""P2 — Finalize (a)/(b) on full pools (pool_en + HN).

Reads estimator_scores.csv from the Reason re-score run. No server calls.
Bonferroni k=4 for McNemar: 2 platforms × 2 thresholds (τ=0.9, 0.99).
"""

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
sys.path.insert(0, str(ROOT / "src"))

from esrc.metrics_calibration import (  # noqa: E402
    bonferroni_alpha,
    bootstrap_metric_ci,
    brier_score,
    evaluate_scores,
    expected_calibration_error,
    mcnemar_threshold_match,
    recall_at_precision,
    reliability_table,
)
from esrc.paths import summer_root  # noqa: E402

N_BOOT = 10_000
SEED = 2026
ALPHA = 0.05
# 2 platforms × 2 thresholds
N_MCNEMAR_TESTS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize full-pool P2 (a)/(b) tables")
    p.add_argument(
        "--scores-dir",
        type=str,
        default=str(summer_root() / "results" / "runs" / "p2_full_pool_ab_rescore"),
        help="Dir containing pool_en/ and hn/ estimator_scores.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(summer_root() / "results" / "p2_confidence" / "tables_full_pool"),
    )
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def as_bool(v: object) -> bool:
    return str(v).lower() in {"1", "true", "yes"}


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and (x != x):
        return ""
    return f"{x:.{digits}f}"


def load_ok_scores(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    # last-wins per user
    by: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = r.get("query_user_id")
            if uid:
                by[uid] = r
    return [r for r in by.values() if r.get("status") == "ok"]


def analyze_pool(
    pool: str,
    rows: list[dict],
    *,
    n_boot: int,
    seed: int,
    alpha_adj: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    y = [as_bool(r["correct"]) for r in rows]
    score_a = [float(r["score_a"]) for r in rows]
    score_b = [float(r["score_b"]) for r in rows]
    n_pos = sum(y)

    summary_rows = []
    for label, conf in [
        ("a_verbalized", score_a),
        ("b_selected_id_exp_logprob", score_b),
    ]:
        m = evaluate_scores(conf, y)
        summary_rows.append(
            {
                "pool": pool,
                "estimator": label,
                "n": m.n,
                "n_correct": m.n_correct,
                "top1_accuracy": fmt(m.top1_accuracy, 4),
                "ece": fmt(m.ece, 4),
                "brier": fmt(m.brier, 4),
                "average_precision": fmt(m.average_precision, 4),
                "recall_at_90_precision": fmt(m.recall_at_90_precision, 4),
                "recall_at_99_precision": fmt(m.recall_at_99_precision, 4),
                "note_thin_positives": (
                    "few_positives_expect_noisy_recall_ci"
                    if n_pos < 50
                    else ""
                ),
            }
        )

    def _ap(c, yy):
        yy = np.asarray(yy)
        if yy.sum() == 0:
            return None
        if yy.sum() == len(yy):
            return 1.0
        return float(average_precision_score(yy, c))

    metrics_to_boot = [
        ("recall_at_90_precision", lambda c, yy: recall_at_precision(c, yy, 0.90)),
        ("recall_at_99_precision", lambda c, yy: recall_at_precision(c, yy, 0.99)),
        ("ece", lambda c, yy: float(expected_calibration_error(c, yy))),
        ("brier", lambda c, yy: float(brier_score(c, yy))),
        ("average_precision", _ap),
    ]

    boot_rows = []
    for label, conf in [
        ("a_verbalized", score_a),
        ("b_selected_id_exp_logprob", score_b),
    ]:
        for metric_name, fn in metrics_to_boot:
            undef_zero = metric_name.startswith("recall_at_")
            ci = bootstrap_metric_ci(
                conf,
                y,
                fn,
                n_boot=n_boot,
                seed=seed,
                undefined_as_zero=undef_zero,
            )
            boot_rows.append(
                {
                    "pool": pool,
                    "estimator": label,
                    "metric": metric_name,
                    "n": len(rows),
                    "n_correct": n_pos,
                    "point": fmt(ci.point, 4),
                    "boot_mean": fmt(ci.mean, 4),
                    "ci95_low": fmt(ci.ci_low, 4),
                    "ci95_high": fmt(ci.ci_high, 4),
                    "n_boot": ci.n_boot,
                    "n_boot_undefined": ci.n_undefined,
                    "undefined_as": ci.undefined_as,
                    "seed": seed,
                }
            )

    mcnemar_rows = []
    for tau in (0.9, 0.99):
        res = mcnemar_threshold_match(score_a, score_b, y, threshold=tau)
        mcnemar_rows.append(
            {
                "pool": pool,
                "comparison": "a_vs_b",
                "threshold": tau,
                "n11_both_match": res.n11,
                "n10_a_only": res.n10,
                "n01_b_only": res.n01,
                "n00_neither": res.n00,
                "n_discordant": res.n_discordant,
                "statistic": fmt(res.statistic, 4),
                "p_value": fmt(res.p_value, 6),
                "alpha": ALPHA,
                "n_tests_bonferroni": N_MCNEMAR_TESTS,
                "alpha_bonferroni": fmt(alpha_adj, 6),
                "significant_bonferroni": str(res.p_value < alpha_adj).lower(),
                "note": res.note,
            }
        )

    rel_rows = []
    for label, conf in [
        ("a_verbalized", score_a),
        ("b_selected_id_exp_logprob", score_b),
    ]:
        for b in reliability_table(conf, y, n_bins=10):
            rel_rows.append(
                {
                    "pool": pool,
                    "estimator": label,
                    "bin_lo": fmt(b.bin_lo, 2),
                    "bin_hi": fmt(b.bin_hi, 2),
                    "n": b.n,
                    "mean_confidence": fmt(b.mean_confidence, 4),
                    "accuracy": fmt(b.accuracy, 4),
                    "gap_abs": fmt(b.gap, 4),
                }
            )

    meta = {
        "pool": pool,
        "n": len(rows),
        "n_correct": n_pos,
        "top1": n_pos / len(rows) if rows else None,
    }
    return summary_rows, boot_rows, mcnemar_rows, rel_rows, meta


def main() -> int:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_adj = bonferroni_alpha(ALPHA, N_MCNEMAR_TESTS)

    all_summary: list[dict] = []
    all_boot: list[dict] = []
    all_mc: list[dict] = []
    all_rel: list[dict] = []
    pool_meta: list[dict] = []

    for pool in ("pool_en", "hn"):
        rows = load_ok_scores(scores_dir / pool / "estimator_scores.csv")
        if not rows:
            raise SystemExit(f"No ok scores for {pool}")
        s, b, m, r, meta = analyze_pool(
            pool, rows, n_boot=args.n_boot, seed=args.seed, alpha_adj=alpha_adj
        )
        all_summary.extend(s)
        all_boot.extend(b)
        all_mc.extend(m)
        all_rel.extend(r)
        pool_meta.append(meta)
        print(
            f"[{pool}] n={meta['n']} correct={meta['n_correct']} "
            f"top1={meta['top1']:.3f}"
        )

    def _write(name: str, rows: list[dict]) -> Path:
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path

    p_sum = _write("table_ab_metrics.csv", all_summary)
    p_boot = _write("table_ab_bootstrap_ci.csv", all_boot)
    p_mc = _write("table_ab_mcnemar.csv", all_mc)
    p_rel = _write("table_ab_reliability.csv", all_rel)

    def _ci(pool: str, est: str, metric: str) -> tuple[str, str, str]:
        for r in all_boot:
            if r["pool"] == pool and r["estimator"] == est and r["metric"] == metric:
                return r["point"], r["ci95_low"], r["ci95_high"]
        return "", "", ""

    def _sum(pool: str, est: str) -> dict:
        for r in all_summary:
            if r["pool"] == pool and r["estimator"] == est:
                return r
        return {}

    lines = [
        "# P2 estimators (a) vs (b) — full-pool deliverable",
        "",
        f"Source: `{scores_dir}`  ",
        f"bootstrap n={args.n_boot}, seed={args.seed}. "
        f"McNemar Bonferroni k={N_MCNEMAR_TESTS} (2 platforms × 2 thresholds), "
        f"α/k={fmt(alpha_adj, 4)}.",
        "",
        "## Summary",
        "",
        "| Pool | Estimator | n | top-1 | ECE | Brier | AP | R@90%P | 95% CI | R@99%P | 95% CI |",
        "|------|-----------|---|-------|-----|-------|-----|--------|--------|--------|--------|",
    ]
    for pool in ("pool_en", "hn"):
        for est, short in [
            ("a_verbalized", "(a) verbalized"),
            ("b_selected_id_exp_logprob", "(b) selected-id exp(lp)"),
        ]:
            s = _sum(pool, est)
            p90, lo90, hi90 = _ci(pool, est, "recall_at_90_precision")
            p99, lo99, hi99 = _ci(pool, est, "recall_at_99_precision")
            lines.append(
                f"| {pool} | {short} | {s.get('n')} | {s.get('top1_accuracy')} | "
                f"{s.get('ece')} | {s.get('brier')} | {s.get('average_precision')} | "
                f"{p90} | [{lo90}, {hi90}] | {p99} | [{lo99}, {hi99}] |"
            )

    lines += [
        "",
        "## McNemar (a) vs (b) — thresholded classifiers",
        "",
        f"| Pool | τ | n10 (a only) | n01 (b only) | disc. | χ² | p | sig @ α/k={fmt(alpha_adj, 4)} |",
        "|------|---|--------------|--------------|-------|----|---|------------------|",
    ]
    for r in all_mc:
        lines.append(
            f"| {r['pool']} | {r['threshold']} | {r['n10_a_only']} | {r['n01_b_only']} | "
            f"{r['n_discordant']} | {r['statistic']} | {r['p_value']} | "
            f"{r['significant_bonferroni']} |"
        )

    # HN thin-data flag
    hn_meta = next(m for m in pool_meta if m["pool"] == "hn")
    pe_meta = next(m for m in pool_meta if m["pool"] == "pool_en")
    # Document non-ok rows left in the re-score artifacts (excluded from metrics).
    fail_note = (
        "Parse failures excluded from denominators (no retry): see run "
        "`reason_predictions.jsonl` / `estimator_scores.csv` for status≠ok."
    )
    fail_ids: list[str] = []
    n_attempted = 0
    for pool in ("pool_en", "hn"):
        path = scores_dir / pool / "estimator_scores.csv"
        if not path.exists():
            continue
        by: dict[str, dict] = {}
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                uid = r.get("query_user_id")
                if uid:
                    by[uid] = r
        n_attempted += len(by)
        for uid, r in sorted(by.items()):
            if r.get("status") != "ok":
                fail_ids.append(f"{pool} `{uid}`")
    if fail_ids:
        fail_note = (
            f"Parse failures excluded from denominators (no retry): "
            f"**{len(fail_ids)}/{n_attempted}** (`error`, truncated JSON) — "
            + "; ".join(fail_ids)
            + "."
        )

    lines += [
        "",
        "## Notes",
        "",
        f"- pool_en: n={pe_meta['n']}, correct={pe_meta['n_correct']} "
        f"({100*pe_meta['top1']:.1f}% top-1).",
        f"- HN: n={hn_meta['n']}, correct={hn_meta['n_correct']} "
        f"({100*hn_meta['top1']:.1f}% top-1). **Expect noisier ECE / R@P CIs** — "
        "fewer positives from the P1 chunking gap, not necessarily a new bug. "
        "Prefer AP / ECE / Brier (+ CIs) over HN Recall@P point estimates.",
        f"- {fail_note}",
        "- (a) and (b) share the same Reason pick; McNemar compares confidence-"
        "threshold classifiers, not pick accuracy.",
        "- **McNemar vs ranking is not a contradiction:** on HN at τ=0.9, "
        "n10=93 > n01=42 ((a)-only exceeds (b)-only at that single threshold), "
        "yet (b) still wins on AP and R@90%P. McNemar at one τ captures local "
        "accept/reject disagreement of the hard classifiers `conf≥τ`; AP / R@P "
        "summarize ranking quality across the full score order. A threshold "
        "where (a)’s mass sits just above τ can flip the discordant counts "
        "without improving discrimination overall.",
        "- Estimator (c) not included (deferred redesign; needs server).",
        "",
        "## Files",
        "",
        "- `table_ab_metrics.csv`",
        "- `table_ab_bootstrap_ci.csv`",
        "- `table_ab_mcnemar.csv`",
        "- `table_ab_reliability.csv`",
        "",
    ]
    md_path = out_dir / "DELIVERABLE_ab_full_pool.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "task": "p2_finalize_ab_full_pool",
        "scores_dir": str(scores_dir),
        "pools": pool_meta,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "bonferroni_k": N_MCNEMAR_TESTS,
        "bonferroni_alpha": alpha_adj,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": [str(p_sum), str(p_boot), str(p_mc), str(p_rel), str(md_path)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote deliverables → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
