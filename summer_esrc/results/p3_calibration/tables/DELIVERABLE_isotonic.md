# P3 isotonic calibration — regression_50 half-split validation

Split seed=42 (P3 cal/test; **not** fixture seed 2026). n_cal=25, n_test=25.

Logic check only — numbers are noisy at n=25; full-pool numbers later.

## Blind τ=0.9 vs isotonic-then-0.9 (test half)

| Estimator | Policy | Accept | Precision | Recall |
|-----------|--------|--------|-----------|--------|
| a_verbalized | naive_raw_0.9 | 4 | 1.0000 | 0.3636 |
| a_verbalized | isotonic_then_0.9 | 0 |  | 0.0000 |
| b_selected_id_exp_logprob | naive_raw_0.9 | 10 | 0.8000 | 0.7273 |
| b_selected_id_exp_logprob | isotonic_then_0.9 | 8 | 0.8750 | 0.6364 |

## Files

- `table_isotonic_threshold_policies.csv`
- `table_isotonic_summary.csv`
- `isotonic_test_scores.csv`
