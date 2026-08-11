# P2 prediction / run schema

## Run directory

`results/runs/<run_id>/`

| File | Role |
|------|------|
| `manifest.json` | model, prompt hash, seed, fixture_id, git commit, input checksums |
| `reason_predictions.jsonl` | one JSON object per query (P2.3 pick + verbalized conf + logprobs meta) |
| `estimator_scores.csv` | per-query scores a/b/c (+ optional c diagnostics) |
| `metrics_by_estimator.csv` | ECE, Brier, AP, Recall@90/99%P |
| `inputs/` | optional cached fixture reason packs (see below) |

## `reason_predictions.jsonl` (one line / query)

```json
{
  "fixture_id": "regression_50",
  "T_level": "T8",
  "query_user_id": "user_…",
  "candidate_user_ids": ["…", "…"],
  "selected_candidate_user_id": "user_…",
  "selected_candidate_number": 3,
  "verbalized_confidence": 0.85,
  "reasoning_short": "…",
  "correct": true,
  "model": "qwen3.6-35b-a3b-nvfp4",
  "enable_thinking": false,
  "sequence_logprob_full": -12.3,
  "selected_id_logprob": -0.02,
  "n_token_logprobs": 40,
  "raw_text": "{…}",
  "status": "ok",
  "error": ""
}
```

## `estimator_scores.csv` columns

- `query_user_id`, `correct`
- `score_a` — verbalized confidence ∈ [0,1]
- `score_b_logprob` — sum logprob of selected-number tokens
- `score_b` — `exp(score_b_logprob)` ∈ (0,1] for ECE/Brier/PR
- `score_c` — softmax mass on P2.3-selected index
- `score_c_argmax_number` — diagnostic only (1-based); not used as primary score
- `c_argmax_disagrees_with_pick` — 1 if argmax(c) ≠ P2.3 pick, else 0
- `candidate_scores_json` — length-15 raw scores used for softmax

`metrics_by_estimator.csv` also reports, on the (c) row only:
`c_argmax_disagree_n`, `c_argmax_disagree_rate`, `score_c_mean`,
`score_c_median`, `score_c_uniform_baseline` (=1/15).

## Identical predictions rule

(a)/(b)/(c) all use the **same** `selected_candidate_*` from the single Reason pick call.
(c) does not re-pick via argmax.
