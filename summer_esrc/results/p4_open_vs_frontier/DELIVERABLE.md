# P4 — Open vs frontier (consolidated deliverable)

**Brief questions:** (C1) How close are open-weight models to gpt-4o? (C2) Where does any gap concentrate — Extract or Reason? (C3) Does the open-weight calibration failure reproduce across stage assignments?

| Claim | Short answer |
|-------|--------------|
| **C1** | Reddit top-1 ≈ gpt-4o; Reddit Hit@15 and all HN metrics lag — HN gap explained by Extract chunking / Search collapse. |
| **C2** | Stage swap (strong Extract / weak Reason vs baseline) → **no significant** top-1/Hit@15 change. Stage assignment is **exchangeable within noise** at n≈500; does **not** support “gap concentrates in Reason.” |
| **C3** | Calibration pattern **reproduces** under cross-tier Reason (`qwen3.5-4b`): verbalized bad, logprob better, HN (a) R@90%P=0. Task property, not one stack’s quirk. |

Stacks:

| | Extract | Reason |
|--|---------|--------|
| Baseline (P1 / C1 local) | `qwen3.5-4b` | `qwen3.6-35b-a3b-nvfp4` |
| Cross-tier (C2 / C3 scores) | `qwen3.6-35b-a3b-nvfp4` | `qwen3.5-4b` |
| Frontier ref (C1) | archived gpt-4o summaries | archived gpt-4o Reason |

---

## C1 — Open-weight vs gpt-4o

Full write-up: [`C1_open_vs_frontier.md`](C1_open_vs_frontier.md). Tables: `tables/`.

| Pool | Metric | Open-weight | gpt-4o | Δ pp [95% CI] | McNemar p |
|------|--------|-------------|--------|---------------|-----------|
| pool_en | top-1 | 36.2% [32.0–40.4] | 39.4% [35.2–43.8] | −3.2 [−7.6, +1.2] | 0.18 |
| pool_en | Hit@15 | 44.0% [39.6–48.4] | 53.8% [49.4–58.2] | **−9.8** [−14.4, −5.2] | 4.5e-5 |
| hn | top-1 | 16.6% [13.4–20.0] | 38.0% [33.6–42.4] | **−21.4** [−26.1, −16.8] | 1.7e-18 |
| hn | Hit@15 | 23.8% [20.0–27.7] | 59.4% [55.0–63.8] | **−35.5** [−40.3, −30.7] | 9.4e-38 |

**Takeaway:** Open weights nearly match frontier on Reddit *decision* accuracy; they lose on *retrieval* (Hit@15) and fail to transfer to HN, where long-profile chunking destroys Search coverage.

---

## C2 — Cross-tier (baseline vs strong-Extract / weak-Reason)

Full write-up: [`C2_cross_tier.md`](C2_cross_tier.md). Tables: `tables_c2/`.

| Pool | Metric | Baseline | Cross-tier | Δ (cross−base) pp [95% CI] | McNemar p |
|------|--------|----------|------------|----------------------------|-----------|
| pool_en | top-1 | 36.2% [32.0–40.4] | 38.7% [34.5–42.9] | +2.4 [−1.8, +6.8] | 0.33 |
| pool_en | Hit@15 | 44.0% [39.6–48.4] | 47.3% [42.9–51.7] | +3.2 [−1.2, +7.8] | 0.19 |
| hn | top-1 | 16.6% [13.4–20.0] | 18.2% [14.8–21.6] | +1.6 [−2.4, +5.6] | 0.48 |
| hn | Hit@15 | 23.8% [20.0–27.7] | 24.2% [20.6–28.1] | +0.4 [−3.8, +4.6] | 0.93 |

All four Δ CIs include 0. Point estimates favor strong Extract, but **not significant**. Chunking drops sharply under 35b Extract (pool_en 75%→18% chunked; HN 99.6%→74%) without a detectable end-to-end accuracy gain — mechanics ≠ attack rate.

**Takeaway:** Within this 4b↔35b pair, reallocating the larger model to Reason (or Extract) does not move top-1/Hit@15 beyond sampling noise. Closing the gpt-4o gap needs better Extract *procedure* or stronger models on both stages, not a Reason-only upgrade story.

---

## C3 — Calibration failure under both stage assignments

Full write-up: [`C3_calibration.md`](C3_calibration.md). Cross-tier scores: `tables_c3_cross_tier/` (from `results/runs/p4_cross_tier_ab_rescore`).

| Stack | Pool | (a) ECE | (a) AP | (a) R@90%P | (b) ECE | (b) AP | (b) R@90%P |
|-------|------|---------|--------|------------|---------|--------|------------|
| Baseline | pool_en | 0.407 | 0.616 | 0.221 | 0.421 | **0.816** | **0.448** |
| Cross-tier | pool_en | 0.399 | 0.660 | 0.333 | **0.352** | **0.825** | **0.578** |
| Baseline | hn | 0.688 | 0.353 | **0.000** | 0.583 | **0.660** | 0.159 |
| Cross-tier | hn | 0.654 | 0.446 | **0.000** | **0.497** | **0.665** | 0.222 |

**Takeaway:** Same qualitative failure on both stacks — verbalized overconfident/poorly ranked; logprob better but still not an operable 90%-precision filter (esp. HN (a)=0). Supports treating calibration limits as **task-level**, not “35b Reason is uniquely miscalibrated.”

---

## Overall summary (answers to P4)

1. **Open vs frontier (C1):** Competitive on Reddit top-1; clearly behind on HN and on Reddit Hit@15. The HN deficit is mechanistic (chunked Extract → weak Search), not unexplained Reason failure.
2. **Where the gap lives (C2):** **Not isolable to Reason** with this swap. Baseline and cross-tier are statistically exchangeable on accuracy. Directionally, stronger Extract helps a little; it does not flip the HN story.
3. **Calibration (C3):** The P2/P3 confidence failure **reproduces** when Reason is the weak 4b model on cross-tier summaries. Abstention / high-precision operation remains fragile under open weights regardless of which stage holds the larger model.

**One-line paper claim:** Local open-weight ESRC nearly matches gpt-4o on Reddit decisions, fails to transfer to HN for Extract/Search reasons, does not improve by parking capacity in Reason vs Extract, and inherits the same confidence-calibration limits in either configuration.
