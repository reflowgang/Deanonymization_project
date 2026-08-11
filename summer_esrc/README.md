# Summer semester — calibration paper (P1–P4)

Owned pipeline + flagship calibration experiments. Lives beside the BSP repo;
does **not** duplicate BSP data or Lermen prompts.

## BSP references (read-only)

From this folder, parent = BSP root:

| Artifact | Path |
|----------|------|
| Prompts | `../prompts/` |
| Reddit pool | `../data/esrc/pool_en/` (and related) |
| Hacker News | `../data/` HN splits / ESRC outputs as used in BSP |
| BSP experiment scripts | `../experiments/esrc/` (reference only; do not edit for summer work) |

## Quick start

```bash
cd summer_esrc
cp .env.example .env   # or export vars
python experiments/p1_divergence/01_vllm_logprob_smoke.py
```

Uses the parent BSP `.venv` or any env with `openai` + `python-dotenv`.
