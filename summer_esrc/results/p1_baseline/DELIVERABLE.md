# P1 hand-in — local open-weight vs archived gpt-4o (T8)

**Models:** Extract `qwen3.5-4b` + Reason `qwen3.6-35b-a3b-nvfp4` vs archived BSP gpt-4o Reason (T8).  
**Denominators:** pool_en 500/500, HN 499/499 (archived HN has 500; paired n=499 excludes `user_0501`).  
**Inference:** bootstrap 10 000 resamples, seed=2026; McNemar exact on paired users.

## Claims

| ID | Claim | Verdict |
|----|--------|---------|
| **C1** | Local stack matches gpt-4o on Reddit (pool_en) | **Mostly supported on top-1**; Hit@15 significantly worse |
| **C2** | Local stack transfers to HN at gpt-4o level | **Rejected** — large, significant gaps on top-1 and Hit@15 |

## Comparison table (95% bootstrap CIs)

| Claim | Pool | Metric | Local | gpt-4o | Δ (local−gpt4o) pp | McNemar p |
|-------|------|--------|-------|--------|--------------------|-----------|
| C1 | pool_en | top-1 | 36.2% [32.0–40.4] | 39.4% [35.2–43.8] | **−3.2** [−7.6, +1.2] | 0.18 |
| C1 | pool_en | Hit@15 | 44.0% [39.6–48.4] | 53.8% [49.4–58.2] | **−9.8** [−14.4, −5.2] | 4.5e-5 |
| C2 | hn | top-1 | 16.6% [13.4–20.0] | 38.0% [33.6–42.4] | **−21.4** [−26.1, −16.8] | 1.7e-18 |
| C2 | hn | Hit@15 | 23.8% [20.0–27.7] | 59.4% [55.0–63.8] | **−35.5** [−40.3, −30.7] | 9.4e-38 |

Machine-readable: `tables/P1_comparison_table.csv`, `tables/local_vs_gpt4o_bootstrap_ci.csv`, `tables/local_vs_gpt4o_mcnemar.csv`.

### Flag — Reddit Hit@15 dropped even though top-1 held

On pool_en, Search is **measurably weaker** (−9.8 pp Hit@15; CI excludes 0) while final Reason top-1 stays statistically compatible with gpt-4o. Same embedder on both sides (`all-mpnet-base-v2`) and the same archived candidate gallery — so this is not a jina-vs-mpnet swap. The gap is in **query-summary content**: local Extract (`qwen3.5-4b`, often chunk-merged) vs archived gpt-4o summaries fed into identical retrieval. Reason still recovers a comparable pick when enough signal remains in the top-15 (or among near neighbors), which is why C1 can pass on top-1 without passing on Hit@15.

## Prediction files

| File | n |
|------|---|
| `predictions/local_pool_en_T8_reason_predictions.jsonl` (+ `.csv`) | 500 |
| `predictions/local_hn_T8_reason_predictions.jsonl` (+ `.csv`) | 499 |

## HN chunking finding (explanation for C2 failure)

See `HN_CHUNKING_FINDING.md`. Bucket sizes (HN, n=499): **2–3: 114**, **4–6: 295**, **7–10: 82**, **11+: 6**. Top-1 falls 20.2% → 16.6% → 13.4% → 0%; the 11+ zero is real but **n=6 only** — treat as the tail of the trend, not a standalone rate. Reddit shows no such chunk–accuracy decline. Search Hit@15 collapse (24% vs 59%) is the main mechanism.
