# P2 estimators (a) vs (b) — regression_50 deliverable

Source run: `/Users/reflowgang/Downloads/deanonymization_project/summer_esrc/results/runs/p2_regression_50_T8/estimator_scores.csv`  
n=50, top-1 correct=23/50, bootstrap n=10000, seed=2026.

## Summary

| Estimator | ECE | Brier | AP | R@90%P | 95% CI | R@99%P | 95% CI |
|-----------|-----|-------|-----|--------|--------|--------|--------|
| (a) verbalized | 0.3980 | 0.3693 | 0.7171 | 0.0000 | [0.0000, 0.7308] | 0.0000 | [0.0000, 0.6538] |
| (b) selected-id exp(logprob) | 0.3530 | 0.2952 | 0.9012 | 0.7826 | [0.2273, 0.9600] | 0.3478 | [0.1739, 0.9130] |

## McNemar (a) vs (b) — thresholded correctness classifiers

Predict “correct” iff conf ≥ τ. Continuity-corrected McNemar; Bonferroni α/m with m=2 (τ=0.9, 0.99).

| τ | n10 (a only) | n01 (b only) | disc. | χ² | p | sig @ α/m=0.0250 |
|---|--------------|--------------|-------|----|---|-------------------------------|
| 0.9 | 6 | 7 | 13 | 0.0000 | 1.000000 | false |
| 0.99 | 1 | 15 | 16 | 10.5625 | 0.001154 | true |

**Note:** (a) and (b) share the same Reason pick; McNemar compares the *confidence-threshold classifiers*, not pick accuracy. n=50 CIs are wide — treat as fixture-scale diagnostics, not pool claims.

## Files

- `table_ab_metrics.csv`
- `table_ab_bootstrap_ci.csv`
- `table_ab_mcnemar.csv`
- `table_ab_reliability.csv`
