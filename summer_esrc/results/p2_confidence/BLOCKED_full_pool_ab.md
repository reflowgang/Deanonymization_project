# P2 full-pool (a)/(b) — BLOCKED pending Reason re-score

## Status

Cannot compute full-pool (a)/(b) from existing P1 overnight files.

P1 `reason_predictions.jsonl` (pool_en 500 / HN 499) are **pick-only**:
`selected_candidate_*`, `correct`, `true_in_top15` — **no**
`verbalized_confidence`, **no** `token_logprobs` / `selected_id_logprob`.

Those fields were only captured in the P2 fixture runner
(`experiments/p2_confidence/02_run_reason_pick.py`), not in
`05_run_full_pool.py`.

Estimator (c) remains deferred (server load); agreed.

## Ready when you approve a Reason-only re-score

Reuses P1 Extract + Search (no re-Extract). ~999 Reason calls with
`logprobs=True`. At ~3 s/user sequential ≈ **45–60 min**.

```bash
cd summer_esrc
export VLLM_BASE_URL=https://gpu6.sedan.pro/v1 VLLM_API_KEY=local-vllm PYTHONUNBUFFERED=1

# 1) Reason re-score (a)/(b) inputs — SERVER
../.venv/bin/python -u experiments/p2_confidence/07_rescore_reason_ab_full_pool.py \
  --pool both --concurrency 1 --timeout 180 --resume \
  --out-dir results/runs/p2_full_pool_ab_rescore

# 2) Offline metrics / McNemar k=4 / reliability — NO SERVER
../.venv/bin/python experiments/p2_confidence/08_finalize_ab_full_pools.py \
  --scores-dir results/runs/p2_full_pool_ab_rescore \
  --out-dir results/p2_confidence/tables_full_pool
```

Fixture-scale (a)/(b) deliverable remains at `tables/` (regression_50).
