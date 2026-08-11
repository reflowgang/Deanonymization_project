# Content-Type Experiment (Step 8)

Tests whether **Personal (P)**, **Opinion (O)**, or **Topical (T)** comment types differ in
deanonymization risk at fixed volume (50 comments).

**Current status:** implementation plan only — no API calls.

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for full workflow, cost estimates,
and script inventory.

## Quick reference

| Label | Meaning |
|-------|---------|
| P | Personal disclosure |
| O | Opinion / value statement |
| T | Topical discussion |

## Data sources

- Query JSONL: `data/raw/POOL-EN/user_*_query.jsonl` (500 users × 500 comments)
- T4 text profiles: `data/esrc/pool_en/truncated_queries/T4/` (50 comments/user)
- Candidates: reuse `data/esrc/pool_en/candidate_summaries/` + embeddings (no classification)

## Classification prompt

`prompts/content_type_classification.txt`

## Recommended approach

1. Classify **full query histories** (not T4-only) in **batches of 10** comments.
2. Build first-50-per-type profiles for qualifying users (≥ 50 of each type).
3. Run ESRC per type (P, O, T) mirroring `experiments/diversity/`.
