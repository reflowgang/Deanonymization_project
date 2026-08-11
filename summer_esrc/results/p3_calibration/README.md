# P3 calibration — deliverable outputs

| Path | Contents |
|------|----------|
| `tables/DELIVERABLE_isotonic.md` | Fixture half-split logic check (n=25; noisy) |
| `tables/table_isotonic_*.csv` | Fixture policy + ECE summary |
| `tables_full_pool/DELIVERABLE_isotonic_full_pool.md` | **Paper path:** full-pool within + cross-platform |
| `tables_full_pool/table_isotonic_full_pool.csv` | All policies, (a)+(b), bootstrap CIs |

Half-splits use **seed=42** (not regression_50 fixture selection seed 2026).  
Script: `experiments/p3_calibration/02_isotonic_full_pool.py` (offline; scores from P2 re-score).
