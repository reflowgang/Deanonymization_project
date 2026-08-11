# P3 isotonic calibration — full-pool deliverable

Scores: `results/runs/p2_full_pool_ab_rescore`  
Half-split seed=**42** (≠ regression_50 fixture seed 2026).  
Requested precision=0.9. Bootstrap n=10000.  
Within-platform: fit on cal half, evaluate blind on test half.  
Cross-platform: fit on **full** source pool, deploy on **full** target.

## Primary table — requested 90% vs delivered (naive vs isotonic)

| Setting | Cal→Test | Estimator | Policy | τ | Survivors | Delivered P | 95% CI | Recall | Note |
|---------|----------|-----------|--------|---|-----------|-------------|--------|--------|------|
| within_platform | pool_en→pool_en | a_verbalized | naive_raw_0.9 | 0.9000 | 20/250 | 0.9500 | [0.8333, 1.0000] | 0.2135 |  |
| within_platform | pool_en→pool_en | a_verbalized | isotonic_thr_for_target_p | 0.9545 | 20/250 | 0.9500 | [0.8333, 1.0000] | 0.2135 |  |
| within_platform | pool_en→pool_en | b_selected_id_exp_logprob | naive_raw_0.9 | 0.9000 | 116/250 | 0.6466 | [0.5603, 0.7321] | 0.8427 |  |
| within_platform | pool_en→pool_en | b_selected_id_exp_logprob | isotonic_thr_for_target_p | 0.7857 | 44/250 | 0.8864 | [0.7826, 0.9750] | 0.4382 |  |
| within_platform | hn→hn | a_verbalized | naive_raw_0.9 | 0.9000 | 36/249 | 0.5556 | [0.3929, 0.7222] | 0.5000 |  |
| within_platform | hn→hn | a_verbalized | isotonic_thr_for_target_p | — | 0/249 | — | — | — | no_cal_threshold_meets_target_precision |
| within_platform | hn→hn | b_selected_id_exp_logprob | naive_raw_0.9 | 0.9000 | 89/249 | 0.3483 | [0.2500, 0.4533] | 0.7750 |  |
| within_platform | hn→hn | b_selected_id_exp_logprob | isotonic_thr_for_target_p | 0.8000 | 33/249 | 0.6061 | [0.4375, 0.7742] | 0.5000 |  |
| cross_platform | pool_en→hn | b_selected_id_exp_logprob | naive_raw_0.9 | 0.9000 | 179/497 | 0.3799 | [0.3099, 0.4503] | 0.8293 |  |
| cross_platform | pool_en→hn | b_selected_id_exp_logprob | isotonic_thr_for_target_p | 0.8182 | 51/497 | 0.7255 | [0.6000, 0.8462] | 0.4512 |  |
| cross_platform | hn→pool_en | b_selected_id_exp_logprob | naive_raw_0.9 | 0.9000 | 223/499 | 0.6547 | [0.5921, 0.7167] | 0.8066 |  |
| cross_platform | hn→pool_en | b_selected_id_exp_logprob | isotonic_thr_for_target_p | 0.8333 | 34/499 | 0.9706 | [0.9000, 1.0000] | 0.1823 |  |

## Cross-platform transfer (estimator b focus; a included in CSV)

See rows with `setting=cross_platform` above (b) and full CSV for (a).

## Operable at 90% precision?

**Direct answer: no — not as a reliable cross-setting attack.** Isotonic on estimator **(b)** can *approach* 90% on Reddit with heavy abstention, but HN within-platform and Reddit→HN transfer miss the bar; naive τ=0.9 is badly over-confident everywhere for (b).

| Claim | Evidence |
|-------|----------|
| Naive raw≥0.9 is **not** 90%-precise | See primary table: Reddit/HN (b) naive delivered P ≪ 0.90 |
| Isotonic helps Reddit (b), still not locked | ~0.89 delivered; CI lower bound &lt; 0.90; ~18% of test queries survive |
| HN within-platform **fails** | (b) isotonic ≪ 0.90; (a) cannot set any cal threshold at 90% P |
| Cross-platform is asymmetric (**C4**) | Reddit→HN misses (0.726). HN→Reddit hits **0.971** [0.900, 1.000] but only **34/499 (~7%)** survivors — transfer is asymmetric, and even the “working” direction is a thin slice, not a practical attacker strategy |
| Reddit (a) naive-0.9 looks precise | **Small-sample artifact** — 20/250 survivors, CI [0.833, 1.000] wide; do **not** cite as “(a) works on Reddit” |

**Practical reading:** abstain-until-confident with (b)+isotonic is a *Reddit-only, low-throughput* filter — **not** operable at 90% on HN, and Reddit calibration does **not** transfer to HN. That framing (abstention filter, not binary attack success) matches what the data support.

## Notes / caveats

- **Reddit (a) at naive τ=0.9 (P=0.95, 20/250):** treat as a **small-sample artifact**, not evidence that verbalized confidence “works” on Reddit. Survivor count is tiny and the CI [0.833, 1.000] is wide; HN (a) cannot even form a 90% threshold.
- **C4 (cross-platform calibration transfer):** Reddit→HN misses the 90% bar. HN→Reddit reaches **0.971** [0.900, 1.000] but only **34/499 (~7%)** queries survive — asymmetric transfer, and even the “success” direction is too thin to be a practical attacker strategy.
- Frame the result as an **abstention filter** (precision bought by discarding most queries), not a binary “attack works / doesn’t.”

## Method notes

- **naive_raw_0.9**: accept if raw conf ≥ 0.9 (no calibration).
- **isotonic_thr_for_target_p**: fit isotonic on cal; choose the lowest threshold on *calibrated* cal scores that achieves ≥ requested precision (max recall among such); apply that τ blind on test calibrated scores.
- **isotonic_then_0.9** / **raw_thr_for_target_p**: extra diagnostics in CSV.
- If cal never reaches requested precision, threshold is undefined → zero survivors (diagnostic failure mode, expected for poorly ranked (a)).
- Bootstrap resamples the **test** set with map/threshold fixed from cal.

## Files

- `table_isotonic_full_pool.csv` — all policies, within + cross, (a)+(b)
- `DELIVERABLE_isotonic_full_pool.md` — this file
- `manifest.json`
