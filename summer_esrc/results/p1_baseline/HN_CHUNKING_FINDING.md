# Finding — HN gap is chunk-merge degradation (not missing users)

## Headline

On full denominators, local top-1 is **16.6%** on HN vs **38.0%** archived gpt-4o (Δ **−21.4** pp, 95% CI excludes 0). The gap is even larger at retrieval: Hit@15 **23.8%** vs **59.4%**. This is a pipeline failure concentrated in **Extract chunk-and-merge**, not Reason noise from incomplete retries.

## Evidence

### 1. Accuracy falls with chunk count (HN only)

From `tables/chunk_audit_by_bucket.csv` (local Reason + local Search). **Ns are the full HN query set (499); say them explicitly so small tails are not over-read:**

| Bucket | HN n | Share | HN top-1 | HN Hit@15 | mean local/gpt-4o char ratio | % truncated (heuristic) |
|--------|------|-------|----------|-----------|------------------------------|-------------------------|
| 2–3 | **114** | 22.8% | 20.2% | 27.2% | 3.2× | 19% |
| 4–6 | **295** | 59.1% | 16.6% | 24.1% | 5.0× | 40% |
| 7–10 | **82** | 16.4% | 13.4% | 19.5% | 6.3× | 65% |
| 11+ | **6** | 1.2% | 0.0% | 16.7% | 7.5× | 83% |

The monotone decline through 2–3 / 4–6 / 7–10 is on **substantial** samples (114 / 295 / 82). The **11+ → 0% top-1 cell is only n=6** — consistent with the trend and useful as a qualitative tail, but **not** a precise rate; write it as “0/6 in the most fragmented profiles,” not as a population percentage.

**pool_en does not show this pattern** (top-1 stays ~34–40% across 1 / 2–3 / 4–6; only 10 users at 7+, of which **1** at 11+). Reddit is 75% chunked but mostly 2–4 chunks; HN is **99.6%** chunked with a long right tail.

### 2. Merge quality proxies track the same axis

As `n_chunks` rises on HN, merged query summaries get longer relative to archived gpt-4o `data/extracted_summaries/T8/*_summary.json`, and mid-generation truncation becomes common (merge `max_tokens=1024` while regurgitating chunk lists). Qualitative side-by-sides (high-N users) show trait dumps, near-duplicate loops, and contradictions — see `chunk_audit/*_COMPARE.md` and the earlier high-N audit under `results/runs/.../hn_chunk_audit/`.

### 3. Failure is upstream of Reason

Hit@15 collapses more than top-1 conditional on retrieval. Noisy query embeddings from bloated/truncated merges pull the true candidate out of the top-15; Reason cannot recover. Candidate-side summaries remain archived (cleaner) → **asymmetric** degradation on the query side only.

## Implication for the write-up

Treat the HN gap as a **positive finding about open-weight Extract limits under long profiles**, not a failed C2 replication of gpt-4o. C1 (Reddit top-1 ≈ gpt-4o) can still stand; C2 fails for a diagnosable reason. Follow-ups: hierarchical merge, higher merge `max_tokens`, hard trait-cap / dedupe in the merge prompt, or skip re-Extract on HN and reuse archived query summaries as an ablation.
