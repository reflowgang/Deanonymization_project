# P1 baseline — deliverable outputs

Final tables/figures for the own-baseline / chunking / full-pool track.

| Path | Contents |
|------|----------|
| `DELIVERABLE.md` | Hand-in one-pager: C1/C2 verdicts + comparison table |
| `HN_CHUNKING_FINDING.md` | Write-up note: HN gap explained by chunk-merge degradation |
| `predictions/` | Local T8 Reason prediction files (pool_en 500, HN 499) |
| `tables/P1_comparison_table.csv` | Compact local vs gpt-4o bootstrap comparison |
| `tables/local_vs_gpt4o_bootstrap_ci.csv` | Full bootstrap + McNemar columns |
| `tables/chunk_audit_by_bucket.csv` | top-1 / Hit@15 / merge proxies by `n_chunks` |
| `chunk_audit/` | Sample side-by-side merge vs gpt-4o summaries |
| `../runs/` | Raw experiment artifacts (do not dig here for paper numbers) |

Reproduce:

```bash
cd summer_esrc
../.venv/bin/python experiments/p1_baseline/06_bootstrap_local_vs_gpt4o.py
../.venv/bin/python experiments/p1_baseline/07_chunk_quality_audit.py
```
