# Diversity Experiment (Step 7)

Fixed text volume **T4 = 50 comments** on Reddit / POOL-EN. Tests whether users with
more diverse subreddit participation are more deanonymizable at equal comment count.

## Group definitions

| Group  | Unique subreddits in first 50 query comments |
|--------|-----------------------------------------------|
| low    | 1–3                                           |
| medium | 4–10                                          |
| high   | 11+                                           |

## Script pipeline

Run from project root:

```bash
# No API
.venv/bin/python3 experiments/diversity/01_assign_diversity_groups.py
.venv/bin/python3 experiments/diversity/02_copy_group_profiles.py

# Reuse main T4 summaries (identical T4 text; no Extract API):
.venv/bin/python3 experiments/diversity/03b_link_main_t4_summaries.py

# No API
.venv/bin/python3 experiments/diversity/04_build_group_embeddings.py
.venv/bin/python3 experiments/diversity/05_search_faiss.py

# API (gpt-4o-mini Reason)
.venv/bin/python3 experiments/diversity/06_reason_top15.py

# No API
.venv/bin/python3 experiments/diversity/07_calibrate_precision_recall.py
.venv/bin/python3 experiments/diversity/08_plot_results.py
```

## Reused main-pipeline artifacts

- Candidate summaries: `data/esrc/pool_en/candidate_summaries/`
- Candidate embeddings: `data/esrc/pool_en/embeddings/candidate_embeddings.npy`
- Candidate index: `results/tables/pool_en_candidate_embeddings_index.csv`

## Outputs

- `results/tables/diversity_group_manifest.csv`
- `results/tables/diversity_reason_predictions.csv`
- `results/tables/diversity_recall_at_precision.csv`
- `results/tables/diversity_precision_recall_curve.csv`
- `results/tables/diversity_summary.csv`
- `results/figures/diversity_recall_at_90.png`
- `results/figures/diversity_top1_accuracy.png`
