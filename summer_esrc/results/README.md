# results/ layout

```
results/
  runs/                  # raw experiment artifacts (run IDs, logs, jsonl) — not for paper tables
  p1_baseline/           # deliverable tables/figures for P1
  p2_confidence/         # deliverable tables/figures for P2
  p3_calibration/        # deliverable tables/figures for P3
  p4_open_vs_frontier/   # deliverable tables/figures for P4
```

Each `pN_*/` has `tables/`, `figures/`, and a short `README.md`.  
Open one folder per task for finished numbers; dig into `runs/` only for debugging.
