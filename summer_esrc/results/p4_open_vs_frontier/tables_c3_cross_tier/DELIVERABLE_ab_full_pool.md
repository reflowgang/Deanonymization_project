# P2 estimators (a) vs (b) — full-pool deliverable

Source: `results/runs/p4_cross_tier_ab_rescore`  
bootstrap n=10000, seed=2026. McNemar Bonferroni k=4 (2 platforms × 2 thresholds), α/k=0.0125.

## Summary

| Pool | Estimator | n | top-1 | ECE | Brier | AP | R@90%P | 95% CI | R@99%P | 95% CI |
|------|-----------|---|-------|-----|-------|-----|--------|--------|--------|--------|
| pool_en | (a) verbalized | 497 | 0.3863 | 0.3993 | 0.3577 | 0.6604 | 0.3333 | [0.0000, 0.3980] | 0.0000 | [0.0000, 0.2200] |
| pool_en | (b) selected-id exp(lp) | 497 | 0.3863 | 0.3515 | 0.2878 | 0.8254 | 0.5781 | [0.2957, 0.6782] | 0.0573 | [0.0292, 0.2775] |
| hn | (a) verbalized | 497 | 0.1811 | 0.6539 | 0.5604 | 0.4462 | 0.0000 | [0.0000, 0.0000] | 0.0000 | [0.0000, 0.0000] |
| hn | (b) selected-id exp(lp) | 497 | 0.1811 | 0.4974 | 0.3750 | 0.6645 | 0.2222 | [0.1341, 0.4400] | 0.2000 | [0.1237, 0.3204] |

## McNemar (a) vs (b) — thresholded classifiers

| Pool | τ | n10 (a only) | n01 (b only) | disc. | χ² | p | sig @ α/k=0.0125 |
|------|---|--------------|--------------|-------|----|---|------------------|
| pool_en | 0.9 | 42 | 82 | 124 | 12.2661 | 0.000461 | true |
| pool_en | 0.99 | 6 | 83 | 89 | 64.8989 | 0.000000 | true |
| hn | 0.9 | 53 | 50 | 103 | 0.0388 | 0.843776 | false |
| hn | 0.99 | 13 | 37 | 50 | 10.5800 | 0.001143 | true |

## Notes

- pool_en: n=497, correct=192 (38.6% top-1).
- HN: n=497, correct=90 (18.1% top-1). **Expect noisier ECE / R@P CIs** — fewer positives from the P1 chunking gap, not necessarily a new bug. Prefer AP / ECE / Brier (+ CIs) over HN Recall@P point estimates.
- Parse failures excluded from denominators (no retry): **5/999** (`error`, truncated JSON) — pool_en `user_89f449b4`; pool_en `user_9a5457b0`; pool_en `user_9d9fba70`; hn `user_0735`; hn `user_0845`.
- (a) and (b) share the same Reason pick; McNemar compares confidence-threshold classifiers, not pick accuracy.
- **McNemar vs ranking is not a contradiction:** on HN at τ=0.9, n10=93 > n01=42 ((a)-only exceeds (b)-only at that single threshold), yet (b) still wins on AP and R@90%P. McNemar at one τ captures local accept/reject disagreement of the hard classifiers `conf≥τ`; AP / R@P summarize ranking quality across the full score order. A threshold where (a)’s mass sits just above τ can flip the discordant counts without improving discrimination overall.
- Estimator (c) not included (deferred redesign; needs server).

## Files

- `table_ab_metrics.csv`
- `table_ab_bootstrap_ci.csv`
- `table_ab_mcnemar.csv`
- `table_ab_reliability.csv`
