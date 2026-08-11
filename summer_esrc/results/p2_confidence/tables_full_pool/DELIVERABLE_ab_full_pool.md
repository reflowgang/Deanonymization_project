# P2 estimators (a) vs (b) — full-pool deliverable

Source: `results/runs/p2_full_pool_ab_rescore`  
bootstrap n=10000, seed=2026. McNemar Bonferroni k=4 (2 platforms × 2 thresholds), α/k=0.0125.

## Summary

| Pool | Estimator | n | top-1 | ECE | Brier | AP | R@90%P | 95% CI | R@99%P | 95% CI |
|------|-----------|---|-------|-----|-------|-----|--------|--------|--------|--------|
| pool_en | (a) verbalized | 499 | 0.3627 | 0.4071 | 0.3566 | 0.6162 | 0.2210 | [0.0000, 0.2825] | 0.0000 | [0.0000, 0.2500] |
| pool_en | (b) selected-id exp(lp) | 499 | 0.3627 | 0.4210 | 0.3467 | 0.8162 | 0.4475 | [0.3315, 0.5879] | 0.0663 | [0.0351, 0.3876] |
| hn | (a) verbalized | 497 | 0.1650 | 0.6877 | 0.6004 | 0.3530 | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] |
| hn | (b) selected-id exp(lp) | 497 | 0.1650 | 0.5827 | 0.4588 | 0.6600 | 0.1585 | [0.0494, 0.3956] | 0.0976 | [0.0430, 0.2644] |

## McNemar (a) vs (b) — thresholded classifiers

| Pool | τ | n10 (a only) | n01 (b only) | disc. | χ² | p | sig @ α/k=0.0125 |
|------|---|--------------|--------------|-------|----|---|------------------|
| pool_en | 0.9 | 75 | 106 | 181 | 4.9724 | 0.025755 | false |
| pool_en | 0.99 | 14 | 92 | 106 | 55.9340 | 0.000000 | true |
| hn | 0.9 | 93 | 42 | 135 | 18.5185 | 0.000017 | true |
| hn | 0.99 | 24 | 47 | 71 | 6.8169 | 0.009030 | true |

## Notes

- pool_en: n=499, correct=181 (36.3% top-1).
- HN: n=497, correct=82 (16.5% top-1). **Expect noisier ECE / R@P CIs** — fewer positives from the P1 chunking gap, not necessarily a new bug. Prefer AP / ECE / Brier (+ CIs) over HN Recall@P point estimates.
- Parse failures excluded from denominators (no retry): **3/999** (`error`, truncated JSON) — pool_en `user_f27da10a`; HN `user_0084`, `user_0792`. Logged in run `reason_predictions.jsonl` / `estimator_scores.csv`.
- (a) and (b) share the same Reason pick; McNemar compares confidence-threshold classifiers, not pick accuracy.
- **McNemar vs ranking is not a contradiction:** on HN at τ=0.9, n10=93 > n01=42 ((a)-only exceeds (b)-only at that single threshold), yet (b) still wins on AP and R@90%P. McNemar at one τ captures local accept/reject disagreement of the hard classifiers `conf≥τ`; AP / R@P summarize ranking quality across the full score order. A threshold where (a)’s mass sits just above τ can flip the discordant counts without improving discrimination overall.
- Estimator (c) not included (deferred redesign; needs server).

## Files

- `table_ab_metrics.csv`
- `table_ab_bootstrap_ci.csv`
- `table_ab_mcnemar.csv`
- `table_ab_reliability.csv`
