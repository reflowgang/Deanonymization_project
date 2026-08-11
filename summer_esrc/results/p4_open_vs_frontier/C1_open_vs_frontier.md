# P4 — C1: open-weight vs frontier (gpt-4o)

**Claim:** How close do the open-weight Extract→Search→Reason stacks get to archived frontier (gpt-4o) Reason at T8?

**Open-weight stack (baseline / P1):** Extract=`qwen3.5-4b`, Reason=`qwen3.6-35b-a3b-nvfp4`.  
**Frontier reference:** archived BSP gpt-4o Reason + FAISS tables (T8).  
**Denominators:** pool_en 500/500 local, 500 gpt-4o; HN 499/499 local (paired n=499; archived HN has 500, excludes `user_0501`).  
**Inference:** bootstrap 10 000 resamples, seed=2026; McNemar exact on paired users. Δ = local − gpt-4o.

*This is the P1 bootstrap comparison, reframed as P4 C1. Numbers unchanged.*

## Comparison table (95% bootstrap CIs)

| Pool | Metric | Open-weight (local) | gpt-4o | Δ (local−gpt4o) pp | McNemar p |
|------|--------|---------------------|--------|--------------------|-----------|
| pool_en | top-1 | 36.2% [32.0–40.4] | 39.4% [35.2–43.8] | **−3.2** [−7.6, +1.2] | 0.18 |
| pool_en | Hit@15 | 44.0% [39.6–48.4] | 53.8% [49.4–58.2] | **−9.8** [−14.4, −5.2] | 4.5e-5 |
| hn | top-1 | 16.6% [13.4–20.0] | 38.0% [33.6–42.4] | **−21.4** [−26.1, −16.8] | 1.7e-18 |
| hn | Hit@15 | 23.8% [20.0–27.7] | 59.4% [55.0–63.8] | **−35.5** [−40.3, −30.7] | 9.4e-38 |

Machine-readable: `tables/P1_comparison_table.csv`, `tables/local_vs_gpt4o_bootstrap_ci.csv`, `tables/local_vs_gpt4o_mcnemar.csv`.

## Verdict

| Platform | Finding |
|----------|---------|
| **Reddit (pool_en)** | Top-1 **compatible** with gpt-4o (Δ CI includes 0). Hit@15 **significantly worse** (−9.8 pp) — Search/query-summary gap under identical `all-mpnet` retrieval, not a Reason-pick collapse. |
| **HN** | **Clear gap** on both metrics. Mechanism: long-profile Extract chunking → weaker query summaries → Hit@15 collapse (24% vs 59%); Reason cannot recover what Search never retrieved. See P1 `HN_CHUNKING_FINDING.md`. |

**C1 answer:** Open-weight models nearly match frontier on Reddit top-1, but trail on retrieval quality; on HN they do **not** transfer to gpt-4o level, for a diagnosable Extract/chunking reason rather than a mysterious Reason deficit.
