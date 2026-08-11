# P2 estimator (c) redesign — full-pool deliverable (C1 + C2-top3)

Source: `results/runs/p2_c_redesign_full_pool` (reuses P2 a/b Reason picks + P1 Extract/Search).  
Model: `qwen3.6-35b-a3b-nvfp4`, thinking off. Bootstrap 10k, seed=2026.  
**Old softmax-15 (c) not re-run** on full pool (15× cost); fixture comparison remains in `tables_c_redesign/`.

## Coverage

- pool_en: **499/499** ok
- HN: **497/497** ok (30 transient 500s retried successfully)

## Metrics

| Pool | Estimator | n | mean✓/mean✗ | ECE | ECE 95% CI | Brier | Brier 95% CI | AP | AP 95% CI | R@90%P | R@90%P 95% CI |
|------|-----------|---|-------------|-----|------------|-------|--------------|----|-----------|--------|---------------|
| pool_en | (a) verbalized | 499 | 0.8522/0.7223 | 0.4071 | [0.3720, 0.4484] | 0.3566 | [0.3313, 0.3824] | 0.6162 | [0.5573, 0.6700] | 0.2210 | [0.0000, 0.2825] |
| pool_en | (b) selected-id exp(lp) | 499 | 0.9338/0.6983 | 0.4210 | [0.3848, 0.4572] | 0.3467 | [0.3170, 0.3768] | 0.8162 | [0.7615, 0.8639] | 0.4475 | [0.3315, 0.5879] |
| pool_en | (c′) C1 P(YES) | 499 | 0.4315/0.2060 | 0.1160 | [0.0927, 0.1621] | 0.2077 | [0.1831, 0.2328] | 0.6073 | [0.5335, 0.6793] | 0.0552 | [0.0223, 0.1179] |
| pool_en | (c′) C2 mean P(A) top-3 | 499 | 0.9530/0.8702 | 0.5375 | [0.4981, 0.5766] | 0.4892 | [0.4554, 0.5229] | 0.7508 | [0.6848, 0.8100] | 0.2762 | [0.0351, 0.4586] |
| hn | (a) verbalized | 497 | 0.8927/0.8448 | 0.6877 | [0.6567, 0.7192] | 0.6004 | [0.5770, 0.6240] | 0.3530 | [0.2632, 0.4509] | 0.0000 | [0.0000, 0.0000] |
| hn | (b) selected-id exp(lp) | 497 | 0.9441/0.7089 | 0.5827 | [0.5531, 0.6127] | 0.4588 | [0.4304, 0.4869] | 0.6600 | [0.5549, 0.7577] | 0.1585 | [0.0494, 0.3956] |
| hn | (c′) C1 P(YES) | 497 | 0.6074/0.3683 | 0.2428 | [0.2140, 0.2779] | 0.1975 | [0.1799, 0.2157] | 0.4280 | [0.3288, 0.5341] | 0.0610 | [0.0147, 0.1538] |
| hn | (c′) C2 mean P(A) top-3 | 497 | 0.9128/0.8375 | 0.6851 | [0.6550, 0.7179] | 0.5988 | [0.5722, 0.6256] | 0.4270 | [0.3261, 0.5477] | 0.0122 | [0.0000, 0.1341] |

## Verdict

- **C1 restores separable confirm/reject** on full pool: Reddit Δ(mean✓−mean✗)=0.226; HN Δ=0.239 (still positive but thinner, consistent with weaker HN picks).
- **C1 calibration** on Reddit: ECE=0.1160 vs (b) ECE=0.4210 — C1 often calibrates better; **ranking still favors (b)** (Reddit AP 0.8162 vs C1 0.6073).
- **C2** stays high-mean (pick usually beats top-3 rivals) — useful margin, still overconfident as a probability.
- Full-pool results **agree with the regression_50 bonus**: old flat (c) was mostly prompt design; C1/C2 fix that; they do **not** dethrone (b) for ranking / R@P.
- HN R@P CIs remain wide — prefer AP/ECE/Brier there.

## Files

- `table_c_redesign_full_pool_metrics.csv`
- `table_c_redesign_full_pool_bootstrap.csv`
- run: `../../runs/p2_c_redesign_full_pool/`
