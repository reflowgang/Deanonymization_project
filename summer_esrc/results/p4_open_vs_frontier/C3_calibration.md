# P4 — C3: does the calibration failure reproduce under the cross-tier stack?

**Claim:** The P2/P3 pattern (verbalized confidence bad, selected-id logprob better, high-precision recall thin / near-zero on HN for (a)) is a **task property**, not an artifact of putting the strong model on Reason.

**Design:** Re-score Reason with verbalized confidence + token logprobs on the **cross-tier** Extract/Search artifacts (strong Extract=`qwen3.6-35b-a3b-nvfp4`, weak Reason=`qwen3.5-4b`), using the same (a)/(b) estimators as P2. Compare to baseline-stack P2 full-pool results (weak Extract / strong Reason).

**Runs:**
- Baseline (a)/(b): `results/runs/p2_full_pool_ab_rescore` → `results/p2_confidence/tables_full_pool/`
- Cross-tier (a)/(b): `results/runs/p4_cross_tier_ab_rescore` → `tables_c3_cross_tier/`
- Script: `experiments/p2_confidence/07_rescore_reason_ab_full_pool.py` + `08_finalize_ab_full_pools.py`

## Side-by-side (a)/(b) metrics

| Stack | Pool | Est | n | top-1 | ECE ↓ | AP ↑ | R@90%P ↑ |
|-------|------|-----|---|-------|-------|------|----------|
| Baseline (4b Ext / 35b Rea) | pool_en | (a) verbalized | 499 | 0.363 | 0.407 | 0.616 | 0.221 |
| Baseline | pool_en | **(b) exp(lp)** | 499 | 0.363 | 0.421 | **0.816** | **0.448** |
| Cross-tier (35b Ext / 4b Rea) | pool_en | (a) verbalized | 497 | 0.386 | 0.399 | 0.660 | 0.333 |
| Cross-tier | pool_en | **(b) exp(lp)** | 497 | 0.386 | **0.352** | **0.825** | **0.578** |
| Baseline | hn | (a) verbalized | 497 | 0.165 | 0.688 | 0.353 | **0.000** |
| Baseline | hn | **(b) exp(lp)** | 497 | 0.165 | 0.583 | **0.660** | 0.159 |
| Cross-tier | hn | (a) verbalized | 497 | 0.181 | 0.654 | 0.446 | **0.000** |
| Cross-tier | hn | **(b) exp(lp)** | 497 | 0.181 | **0.497** | **0.665** | 0.222 |

Bootstrap CIs for cross-tier: `tables_c3_cross_tier/table_ab_bootstrap_ci.csv` (n_boot=10000, seed=2026). Baseline CIs: `../p2_confidence/tables_full_pool/table_ab_bootstrap_ci.csv`.

## Qualitative pattern check

| Pattern (from P2/P3) | Baseline | Cross-tier | Reproduces? |
|----------------------|----------|------------|-------------|
| (a) verbalized badly calibrated (ECE ≫ 0) | Yes (0.41 / 0.69) | Yes (0.40 / 0.65) | **Yes** |
| (b) ranks better than (a) (AP) | Yes | Yes | **Yes** |
| HN (a) R@90%P = 0 (cannot hold 90% precision) | Yes | Yes | **Yes** |
| Even best estimator: R@90%P far from a reliable high-precision attack | Yes (≤0.45 Reddit; ≤0.16 HN) | Yes (≤0.58 Reddit; ≤0.22 HN) | **Yes** |
| Naive confidence ≠ operable 90%-precision selector | Yes (P3 isotonic) | Same score pathology; isotonic not re-fit | **Yes (qualitative)** |

Cross-tier (b) is numerically a bit *cleaner* (lower ECE, higher AP/R@90%P) — consistent with stronger Extract feeding Reason — but the **ordering and failure modes are unchanged**.

## Verdict

**C3 supported.** Calibration failure under open weights is not tied to “strong Reason / weak Extract.” Swapping tiers keeps the same story: verbalized confidence is a poor reliability signal; token-logprob confidence ranks better; neither delivers a trustworthy high-precision (Recall@90%P) attack, especially on HN. Treat this as a **property of the ESRC attack setting**, not a quirk of one stage’s model size.
