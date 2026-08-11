# P2 estimator (c) redesign — regression_50 (C1 + C2-top3)

Source: `results/runs/p2_c_redesign_regression_50/` (reuses P2.3 Reason picks).  
Model: `qwen3.6-35b-a3b-nvfp4`, thinking off. n=50/50 ok. Bootstrap 5k, seed=2026.

## What changed

- **Old (c):** 15 independent forced “is candidate *k* the match?” → softmax mass on pick (~uniform ≈1/15).
- **C1:** binary YES/NO confirm of the pick → `P(YES)`.
- **C2:** pairwise pick-vs-rival for **top-3** Search rivals → `mean P(A)`.

## Metrics (same picks as a/b)

| Estimator | mean | mean✓ | mean✗ | ECE | Brier | AP | R@90%P | R@90%P 95% CI |
|-----------|------|-------|-------|-----|-------|-----|--------|---------------|
| (a) verbalized | 0.8580 | 0.9065 | 0.8167 | 0.3980 | 0.3693 | 0.7171 | 0.0000 | [0.0000, 0.7308] |
| (b) selected-id exp(lp) | 0.8130 | 0.9507 | 0.6957 | 0.3530 | 0.2952 | 0.9012 | 0.7826 | [0.2307, 0.9600] |
| (c) old softmax-15 | 0.0752 | 0.0806 | 0.0706 | 0.3848 | 0.3916 | 0.8488 | 0.6087 | [0.1000, 0.8571] |
| (c′) C1 P(YES) | 0.3959 | 0.5547 | 0.2605 | 0.1428 | 0.1964 | 0.7931 | 0.4783 | [0.1429, 0.7059] |
| (c′) C2 mean P(A) top-3 | 0.8760 | 0.9267 | 0.8328 | 0.4160 | 0.3820 | 0.8220 | 0.4783 | [0.1200, 0.7826] |

## Verdict

- **Old (c) was largely a prompt artifact:** mean≈0.075≈1/15, almost no dynamic range (correct vs incorrect barely differ). Any non-trivial AP there is from tiny within-softmax noise, not usable confidence.
- **C1 fixes the flatness:** mean✓−mean✗ = 0.294 (0.555 vs 0.261); best ECE/Brier of the set (0.143 / 0.196). Separable confirm/reject signal exists once the prompt is binary.
- **C2 also separates** (Δ=0.094) but stays high-mean (0.876) — pick usually beats top-3 rivals even when wrong; useful margin, still overconfident as a probability.
- **(b) remains the ranking champion** on this fixture (AP=0.901, R@90%P=0.783). C1/C2 beat old (c) on calibration/spread; they do **not** dethrone (b).
- n=50 → CIs wide; bonus evidence that (c)’s earlier failure was mostly prompt design, not “no signal in the model.”

## Files

- `table_c_redesign_metrics.csv`
- `table_c_redesign_bootstrap.csv`
- run: `../../runs/p2_c_redesign_regression_50/`
