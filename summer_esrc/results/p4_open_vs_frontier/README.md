# P4 open vs frontier — deliverable outputs

| Path | Contents |
|------|----------|
| **`DELIVERABLE.md`** | **Consolidated P4 hand-in (C1+C2+C3 + overall summary)** |
| `C1_open_vs_frontier.md` | Open-weight vs gpt-4o (P1 bootstrap, reframed) |
| `C2_cross_tier.md` | Baseline vs cross-tier accuracy bootstrap |
| `C3_calibration.md` | (a)/(b) calibration pattern: baseline vs cross-tier |
| `tables/` | C1 bootstrap / McNemar CSVs (local vs gpt-4o) |
| `tables_c2/` | C2 bootstrap / McNemar CSVs (baseline vs cross-tier) |
| `tables_c3_cross_tier/` | Cross-tier (a)/(b) metrics + bootstrap CIs |
| `predictions/` | Baseline local Reason predictions (P1) |

## Stacks

| Role | Baseline (P1 / P4-C1) | Cross-tier (P4-C2 / C3) |
|------|----------------------|------------------------|
| Extract | `qwen3.5-4b` (16k) | `qwen3.6-35b-a3b-nvfp4` (32k) |
| Reason | `qwen3.6-35b-a3b-nvfp4` (32k) | `qwen3.5-4b` (16k) |

## Key runs

| Run | Path |
|-----|------|
| Baseline E→S→R | `results/runs/p1_full_pool_overnight_20260805/` |
| Cross-tier E→S→R | `results/runs/p4_cross_tier_full_pool/` |
| Baseline (a)/(b) rescore | `results/runs/p2_full_pool_ab_rescore/` |
| Cross-tier (a)/(b) rescore | `results/runs/p4_cross_tier_ab_rescore/` |

## Scripts

```bash
# C2 bootstrap
../.venv/bin/python experiments/p4_open_vs_frontier/01_bootstrap_baseline_vs_cross_tier.py

# C3: Reason (a)/(b) on cross-tier Extract/Search
../.venv/bin/python experiments/p2_confidence/07_rescore_reason_ab_full_pool.py \
  --p1-run-dir results/runs/p4_cross_tier_full_pool \
  --out-dir results/runs/p4_cross_tier_ab_rescore \
  --model qwen3.5-4b --concurrency 2 --timeout 300 --resume

../.venv/bin/python experiments/p2_confidence/08_finalize_ab_full_pools.py \
  --scores-dir results/runs/p4_cross_tier_ab_rescore \
  --out-dir results/p4_open_vs_frontier/tables_c3_cross_tier
```
