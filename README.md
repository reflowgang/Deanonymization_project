# Deanonymization project

Local open-weight ESRC (Extract → Search → Reason → Calibrate) experiments
for the summer calibration paper (P1–P4), plus archived companion-paper
artifacts from *How Much Text Is Enough?* (BSP truncation / content-type /
diversity).

| Area | Path |
|------|------|
| Summer pipeline + experiments | `summer_esrc/` |
| Companion BSP scripts | `experiments/` |
| Lermen prompts (shared, read-only for summer) | `prompts/` |
| Companion published tables/figures | `results/` |
| Datasets (truncated queries, summaries, embeddings) | `data/` |

**Default stack (P1 baseline):** Extract `qwen3.5-4b`, Reason `qwen3.6-35b-a3b-nvfp4`,
Search `all-mpnet-base-v2`. Config: `summer_esrc/configs/models.yaml`.
Seeds: **2026** (bootstrap / fixtures), **42** (P3 isotonic cal/test split only).

---

## Metrics

| Label | Meaning |
|-------|---------|
| **Reason top-1** | Final LLM pick equals true user |
| **Hit@15** | True user appears in Search top-15 |
| **Search@1** | True user is rank-1 retrieved (not Reason top-1) |
| **(a)** | Verbalized confidence in JSON |
| **(b)** | `exp(logprob)` of selected candidate-number token |
| **R@90%P / R@99%P** | Max recall at ≥90% / ≥99% precision |

Denominators: pool_en ≈500, HN ≈499 (exact *n* varies slightly when Reason parse fails).

---

## P1 — Local open-weight vs archived gpt-4o (T8)

**Verdict:** Reddit top-1 compatible with gpt-4o; Reddit Hit@15 and all HN metrics lag.
HN mechanism: long-profile Extract chunking → weak query summaries → Search collapse.

| Pool | Metric | Local | gpt-4o | Δ pp [95% CI] | McNemar p |
|------|--------|-------|--------|---------------|-----------|
| pool_en | top-1 | 36.2% [32.0–40.4] | 39.4% [35.2–43.8] | −3.2 [−7.6, +1.2] | 0.18 |
| pool_en | Hit@15 | 44.0% [39.6–48.4] | **53.8%** [49.4–58.2] | **−9.8** [−14.4, −5.2] | 4.5e-5 |
| hn | top-1 | 16.6% [13.4–20.0] | 38.0% [33.6–42.4] | **−21.4** [−26.1, −16.8] | 1.7e-18 |
| hn | Hit@15 | 23.8% [20.0–27.7] | **59.4%** [55.0–63.8] | **−35.5** [−40.3, −30.7] | 9.4e-38 |

Archived gpt-4o Hit@15 is **T8-filtered** FAISS (53.8% / 59.4%). Do not use the
49.2% / 50.5% figures from an early embedder side-check — those mixed T1–T8 rows.

**HN chunking:** accuracy falls with `n_chunks` (2–3→20% top-1 … 11+→0% on n=6 tail).
Reddit shows no such trend. See `summer_esrc/results/p1_baseline/tables/`.

### Erratum — embedder confound

Holding mpnet fixed does **not** imply the Hit@15 gap is Extract-summary quality alone.
Re-embedding the **same** P1 local Extract summaries with `jina-embeddings-v3`:

| Pool | mpnet Hit@15 | jina Hit@15 | Δ [95% CI] |
|------|--------------|-------------|------------|
| pool_en | 44.0% | **66.4%** | **+22.4 pp [+18.0, +27.0]** |
| hn | 23.8% | **45.7%** | **+21.8 pp [+17.2, +26.5]** |

Both factors (summary quality + embedder) contribute. Search-only side check
(no Reason re-run): `summer_esrc/results/x_embedder_check/`.

---

## P2 — Confidence estimators (a) / (b) / (c′)

Full-pool rescore on P1 Extract+Search; *n*=499 pool_en / 497 HN after 3 JSON parse failures.

| Pool | Est | n | top-1 | ECE | Brier | AP | R@90%P | R@99%P |
|------|-----|---|-------|-----|-------|-----|--------|--------|
| pool_en | (a) verbalized | 499 | 0.363 | 0.407 | 0.357 | 0.616 | 0.221 | 0.000 |
| pool_en | **(b) exp(lp)** | 499 | 0.363 | 0.421 | 0.347 | **0.816** | **0.448** | 0.066 |
| hn | (a) verbalized | 497 | 0.165 | 0.688 | 0.600 | 0.353 | **0.000** | 0.000 |
| hn | **(b) exp(lp)** | 497 | 0.165 | 0.583 | 0.459 | **0.660** | 0.159 | 0.098 |

**(b)** ranks better than **(a)**; neither is an operable 90%-precision filter
(esp. HN (a) R@90%P=0). Companion GPT-4o truncation study had Reddit R@90%P ≤2.1%
(max over T1–T8); P2 22.1% is T8 local 35b Reason — real model/study difference, not a typo.

Estimator **(c′)** redesign (forced YES / top-3 margin): C1 calibrates better on Reddit
(ECE 0.116) but does not beat (b) on AP / R@P. Tables:
`summer_esrc/results/p2_confidence/tables_full_pool/`,
`tables_c_redesign_full_pool/`.

---

## P3 — Isotonic calibration (abstention filter)

Half-split seed=**42**. Target precision 0.9. Naive τ=0.9 is badly overconfident for (b).
Isotonic on (b) can *approach* 90% on Reddit with heavy abstention; HN within-platform
fails; Reddit→HN transfer misses; HN→Reddit hits ~0.97 precision but only ~7% survivors.

**Not operable as a reliable cross-setting high-precision attack.**
Tables: `summer_esrc/results/p3_calibration/tables_full_pool/`.

---

## P4 — Open vs frontier / stage swap / Gemma

### Accuracy overview

| Condition | Stack | pool_en top-1 | pool_en Hit@15 | HN top-1 | HN Hit@15 |
|-----------|-------|---------------|----------------|----------|-----------|
| Baseline | 4b Ext / 35b Rea | 36.2% | 44.0% | 16.6% | 23.8% |
| gpt-4o archived | gpt-4o / gpt-4o | 39.4% | 53.8% | 38.0% | 59.4% |
| Cross-tier | 35b Ext / 4b Rea | 38.7% | 47.3% | 18.2% | 24.2% |
| Gemma Reason | 4b Ext / gemma4-26b | 37.8% | 44.8% | 20.0% | 26.9% |
| Jina Search only | same P1 extracts | — | 66.4% | — | 45.7% |

Cross-tier vs baseline: all Δ CIs include 0 (stage assignment exchangeable within noise).
Gemma: directionally better; only HN top-1 Δ CI excludes 0 (McNemar p≈0.05).

### Calibration pattern (reproduces under all stacks)

| Stack | Pool | (a) ECE | (a) AP | (a) R@90%P | (b) ECE | (b) AP | (b) R@90%P |
|-------|------|---------|--------|------------|---------|--------|------------|
| Baseline | pool_en | 0.407 | 0.616 | 0.221 | 0.421 | 0.816 | 0.448 |
| Baseline | hn | 0.688 | 0.353 | 0.000 | 0.583 | 0.660 | 0.159 |
| Cross-tier | pool_en | 0.399 | 0.660 | 0.333 | 0.352 | 0.825 | 0.578 |
| Cross-tier | hn | 0.654 | 0.446 | 0.000 | 0.497 | 0.665 | 0.222 |
| Gemma | pool_en | 0.304 | 0.737 | 0.164 | 0.506 | 0.784 | 0.386 |
| Gemma | hn | 0.600 | 0.497 | 0.000 | 0.643 | 0.643 | 0.270 |

Gemma-specific: ~1.5% of Reason calls need a tolerant JSON parser (markdown fences,
key typo `"selected_candidate_number: N`). Qwen paths keep strict parse.

---

## How to reproduce

```bash
# Env (vLLM on gpu6)
export VLLM_BASE_URL=https://gpu6.sedan.pro/v1
export VLLM_API_KEY=local-vllm
# Optional: VLLM_EXTRACT_MODEL, VLLM_REASON_MODEL

cd summer_esrc
PYTHONPATH=src ../.venv/bin/python experiments/p1_baseline/05_run_full_pool.py \
  --pool both --phase all --resume

# Bootstrap local vs gpt-4o (P1 / P4 C1)
PYTHONPATH=src ../.venv/bin/python experiments/p1_baseline/06_bootstrap_local_vs_gpt4o.py

# P2 (a)/(b) rescore + finalize
PYTHONPATH=src ../.venv/bin/python experiments/p2_confidence/07_rescore_reason_ab_full_pool.py \
  --p1-run-dir results/runs/p1_full_pool_overnight_20260805 \
  --out-dir results/runs/p2_full_pool_ab_rescore --resume
PYTHONPATH=src ../.venv/bin/python experiments/p2_confidence/08_finalize_ab_full_pools.py

# P3 isotonic
PYTHONPATH=src ../.venv/bin/python experiments/p3_calibration/02_isotonic_full_pool.py

# P4 cross-tier / gemma bootstraps
PYTHONPATH=src ../.venv/bin/python experiments/p4_open_vs_frontier/01_bootstrap_baseline_vs_cross_tier.py
PYTHONPATH=src ../.venv/bin/python experiments/p4_open_vs_frontier/02_bootstrap_baseline_vs_gemma_reason.py

# Side check: jina embedder
PYTHONPATH=src ../.venv/bin/python experiments/x_embedder_check/01_run_jina_search_check.py
PYTHONPATH=src ../.venv/bin/python experiments/x_embedder_check/02_bootstrap_mpnet_vs_jina.py
```

Machine-readable tables live under `summer_esrc/results/p{1,2,3,4}_*/` and
`summer_esrc/results/x_embedder_check/`. Run artifacts under `summer_esrc/results/runs/`.

---

## Spring BSP — How Much Text Is Enough?

Bachelor seminar paper (Hlib Petrov; supervisor Radu State; advisor Tatiana Petrova).
Research question: how much text must a user produce before they become
deanonymizable by an LLM-based ESRC attacker? Companion experiments live under
top-level `experiments/`, `prompts/`, `results/`, and `data/` (paths unchanged).

**Stack:** Extract `gpt-4o-mini`, Reason `gpt-4o` (final), Search `all-mpnet-base-v2` + FAISS.
Eight truncation levels T1–T8 (5 → full history comments). Primary attack metric:
**Recall@90% Precision**; MST = smallest T with R@90%P > 5% (noise floor).

### Truncation (main result)

Top-1 rises nearly monotonically with volume; R@90%P stays near zero on both platforms
(Reddit envelope ≤2.1% at T5; HN ≤0.62% at T6). GPT-4o confidence is badly overconfident
(e.g. stated 0.85 → empirical ~26.7% on HN).

| Level | Comments (approx.) | Reddit top-1 [95% CI] | HN top-1 [95% CI] | Reddit R@90%P | HN R@90%P |
|-------|--------------------|------------------------|-------------------|---------------|-----------|
| T1 | 5 | 8.8% [6.4–11.4] | 4.8% [3.0–6.8] | — | — |
| T2 | 10 | 13.4% [10.4–16.4] | 8.6% [6.2–11.0] | — | — |
| T3 | 25 | 16.6% [13.4–20.0] | 15.6% [12.4–18.8] | 1.2% | — |
| T4 | 50 | 25.2% [21.4–29.0] | 22.2% [18.8–25.8] | — | — |
| T5 | 100 | 28.2% [24.4–32.2] | 26.0% [22.2–29.8] | **2.1%** | — |
| T6 | 200 | 33.2% [29.2–37.4] | 32.4% [28.2–36.6] | 1.8% | **0.62%** |
| T7 | 500 | 39.8% [35.6–44.3] | 34.6% [30.4–38.8] | 2.0% | 0.58% |
| T8 | full | 39.4% [35.2–43.6] | 38.0% [33.8–42.2] | 1.0% | 0.53% |

Sources: `results/tables/bootstrap_confidence_intervals.csv`,
`pool_en_recall_at_precision.csv`, `hn_recall_at_precision.csv`.
Scripts: `experiments/esrc/`, `experiments/truncation/`, `experiments/analysis/`.

### Content type (fixed volume, first-50 of each type)

Classify full query histories into **P**ersonal / **O**pinion / **T**opical
(`prompts/content_type_classification.txt`); build first-50-per-type profiles for users
with ≥50 of each type; run ESRC per type. Scripts: `experiments/content_type/`.

| Type | n | top-1 | R@90%P | vs Personal (p) |
|------|---|-------|--------|-----------------|
| Personal | 447 | **32.4%** | 1.43% | — |
| Opinion | 499 | 18.8% | — | 5e-6 |
| Topical | 500 | 18.3% | — | 2e-6 |

Opinion vs Topical: p=0.85 (n.s.). Label mix on ~250k comments: P 18% / O 39% / T 43%.
Tables: `results/tables/content_type_summary.csv`, `content_type_significance_tests.csv`.

### Diversity (fixed T4 = 50 comments)

Groups by unique subreddits in first 50 query comments. Scripts: `experiments/diversity/`.

| Group | Unique subreddits | n | top-1 | Pairwise vs others |
|-------|-------------------|---|-------|--------------------|
| low | 1–3 | 58 | 27.6% | all p>0.05 |
| medium | 4–10 | 179 | 21.2% | all p>0.05 |
| high | 11+ | 263 | 19.0% | all p>0.05 |

Inverse trend (low-diversity more identifiable) is **not significant**.
Tables: `results/tables/diversity_summary.csv`, `diversity_significance_tests.csv`.

### Data layout (`data/`)

| Path | Purpose |
|------|---------|
| `raw/` | Original Reddit dumps / HN exports |
| `filtered/` | Qualifying user ID lists |
| `splits/query_full/`, `splits/candidate/` | Full query / candidate profiles |
| `truncated/T1`…`T8`, `esrc/pool_en/truncated_queries/` | Truncated query profiles |
| `diversity_groups/`, `content_types/` | Experiment subsets |
| `summaries/`, `extracted_summaries/`, `esrc/.../candidate_summaries/` | Extract outputs |
| `embeddings/` | Precomputed embedding matrices |

JSONL comments: fields `user_id`, `text`, `timestamp` (chronological).
Published BSP tables/figures: `results/tables/`, `results/figures/`.
Prompts: `prompts/` (Lermen G.2 summarization, record selection, filtering,
pairwise comparison, content-type classification).
