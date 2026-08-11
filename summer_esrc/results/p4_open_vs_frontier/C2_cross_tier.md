# P4 — C2: cross-tier swap (strong Extract / weak Reason)

**Question:** Does the local-vs-frontier gap concentrate in Reason, or in Extract?

**Design:** Same two models as P1, roles swapped.

| Role | Baseline (P1 / P4-C1) | Cross-tier (P4-C2) |
|------|----------------------|-------------------|
| Extract | `qwen3.5-4b` (16k) | `qwen3.6-35b-a3b-nvfp4` (32k) |
| Reason | `qwen3.6-35b-a3b-nvfp4` (32k) | `qwen3.5-4b` (16k) |

**Runs:** baseline `results/runs/p1_full_pool_overnight_20260805/`; cross-tier `results/runs/p4_cross_tier_full_pool/`.  
**Denominators:** pool_en baseline 500 / cross-tier 499 (1 permanent Reason miss); HN 499/499 both (shared `user_0501` extract failure). Paired n=499 per pool.  
**Inference:** bootstrap 10 000 resamples, seed=2026; McNemar exact on paired users. Δ = cross-tier − baseline.

## Point estimates (full ok sets)

| Pool | Metric | Baseline | Cross-tier | Raw Δ |
|------|--------|----------|------------|-------|
| pool_en | top-1 | 36.2% (181/500) | 38.7% (193/499) | +2.5 pp |
| pool_en | Hit@15 | 44.0% (220/500) | 47.3% (236/499) | +3.3 pp |
| hn | top-1 | 16.6% (83/499) | 18.2% (91/499) | +1.6 pp |
| hn | Hit@15 | 23.8% (119/499) | 24.2% (121/499) | +0.4 pp |

## Bootstrap comparison (95% CIs)

| Pool | Metric | Baseline | Cross-tier | Δ (cross−base) pp | McNemar p | CI excludes 0? |
|------|--------|----------|------------|-------------------|-----------|----------------|
| pool_en | top-1 | 36.2% [32.0–40.4] | 38.7% [34.5–42.9] | **+2.4** [−1.8, +6.8] | 0.33 | No |
| pool_en | Hit@15 | 44.0% [39.6–48.4] | 47.3% [42.9–51.7] | **+3.2** [−1.2, +7.8] | 0.19 | No |
| hn | top-1 | 16.6% [13.4–20.0] | 18.2% [14.8–21.6] | **+1.6** [−2.4, +5.6] | 0.48 | No |
| hn | Hit@15 | 23.8% [20.0–27.7] | 24.2% [20.6–28.1] | **+0.4** [−3.8, +4.6] | 0.93 | No |

Machine-readable: `tables_c2/C2_comparison_table.csv`, `tables_c2/baseline_vs_cross_tier_bootstrap_ci.csv`, `tables_c2/baseline_vs_cross_tier_mcnemar.csv`.  
Recompute: `experiments/p4_open_vs_frontier/01_bootstrap_baseline_vs_cross_tier.py`.

## Verdict

**No statistically supported gain** from moving the strong model onto Extract. All four paired Δ CIs include 0; McNemar never rejects (all p ≥ 0.19).

Directionally, rates move **up** under strong-Extract/weak-Reason (not down), which is the opposite of the pre-registered C2 intuition that any capacity gap would concentrate in Reason. But the lifts (+2–3 pp on Reddit, +1.6 pp HN top-1, ~0 on HN Hit@15) sit inside sampling noise at n≈500.

### Interpretation for the write-up

1. **C2 assumption not confirmed.** We cannot claim Extract quality matters *more* than Reason on these pools — the swap does not produce a detectable improvement (nor a detectable loss).
2. **Also not a Reason-bottleneck story.** Putting the weak model on Reason does not hurt relative to baseline; if anything, point estimates favor the strong-Extract stack. That weakens a simple “Reason is where scale helps” narrative.
3. **HN still looks Extract/Search-bound, but not via this swap.** Cross-tier barely moves HN Hit@15 (+0.4 pp). Chunking remains heavy even with 35b Extract (368/499 chunked vs 91/499 on pool_en) — the long-profile / merge pathology from P1 is not fixed by upgrading Extract alone within this 4b↔35b pair.
4. **Practical takeaway.** Within this open-weight pair, stage assignment is **exchangeable within noise**. Closing the gpt-4o gap (especially HN) likely needs better Extract *procedure* (chunk/merge, context) or stronger models on *both* stages, not just reallocating the larger model to Reason.

## Side note — Extract chunking under the swap

| Pool | Baseline chunked | Cross-tier chunked |
|------|------------------|--------------------|
| pool_en | 375/500 (75%) | 91/499 (18%) |
| hn | 497/499 (99.6%) | 368/499 (74%) |

Stronger Extract + 32k context sharply reduces Reddit chunking and cuts HN chunking, consistent with better single-pass coverage — yet end-to-end top-1/Hit@15 gains remain non-significant. That separates “cleaner Extract mechanics” from “attack accuracy.”
