# Progress — summer_esrc (P1–P4)

Short log of meaningful steps. Newest last.

## 2026-08-04 — Scaffold `summer_esrc/`

Shared pipeline layout (`src/esrc`, `configs`, `experiments/p*`, `results/runs`) as a sibling of the BSP repo, not inside it. BSP data/prompts stay read-only via `../`.

## 2026-08-04 — P1.1 vLLM logprob smoke

Confirmed `https://gpu6.sedan.pro/v1` returns token logprobs for `qwen3.5-4b` via `generate()`. Needed before P2/P3. Run artifact: `results/runs/p1_1_smoke_20260804T121143Z/`.

## 2026-08-04 — Rename `summer_calibration/` → `summer_esrc/`

Top-level rename only; README path updated. Keeps summer work under the agreed name.

## 2026-08-04 — P1.0 BSP artifact inventory

Read-only walk of `../data` and `../results/tables` for pool_en (+ HN pointers). Documented pool sizes, what’s already computed, and where the paper numbers (T8 top-1 39.4%, Hit@15 53.8%, Recall@90%P ≤2.1%) live.

## 2026-08-04 — P1.3 Freeze regression_50 fixture

Stratified 25/25 T8 gpt-4o correct/incorrect, seed=2026, from 497 eligible (excl. 3 missing T7). Wrote `data/fixtures/regression_50/{user_ids.txt,checksums.csv,manifest.json}`. BSP baselines on this set: Hit@15 28/50, top-1 25/50 by construction.

Note: this fixture (seed=2026) is separate from the temperature-scaling cal/test split (seed=42) — do not reuse or mix them for P3.

## 2026-08-04 — Started P2 in parallel with P1 bisection (paused on budget question)

P1.6–P1.12 (teammate divergence bisection vs other students) is **out of scope** in the revised brief, so we stop working on it. P2 (three confidence estimators) continues on regression_50 using the working local-logprob pipeline.

## 2026-08-04 — P2.0 Reason model locked + thinking probe

Locked Reason = `qwen3.6-35b-a3b-nvfp4` in `configs/models.yaml`. Probe on gpu6: default and `chat_template_kwargs.enable_thinking=false` both return JSON content without reasoning traces; `thinking:false` incorrectly *enables* thinking. Pipeline always passes `enable_thinking=False` for Reason and (c) scoring. Design decision: disable thinking for all P2 Reason/(c) calls to avoid 15× CoT blowup.

## 2026-08-04 — P2.1–P2.8 scaffolding (fixture path)

No P1.4/P1.5 Search/Reason artifacts under `results/runs/` (only P1.1 smoke) → fixture inputs from BSP FAISS+summaries. Added schema, `01`–`04` scripts, `confidence`/`metrics_calibration`/`reason_prompt` modules. (b)=selected-number token logprobs only; (c)=softmax mass on P2.3 pick.

## 2026-08-04 — P2 full regression_50 run

Ran 50 Reason picks + ~750 (c) forced scores on `qwen3.6-35b-a3b-nvfp4` (`enable_thinking=False`). Artifacts: `results/runs/p2_regression_50_T8/`. Top-1 23/50. Metrics: (b) best ECE/Brier/AP/R@90P; (a) R@90P=0; (c) mean≈0.075≈uniform, argmax≠pick on 18/50 (36%).

## 2026-08-04 — P2 fixture diagnostics (bootstrap + score_c + disagreement)

Bootstrap 5k: (b) R@90%P point 0.78 but 95%CI [0.23, 0.96] overlaps (a) [0.00, 0.73] — n=50 too wide to claim separation. score_c flat mainly from tiny raw forced-lp spreads (median≈0.35); T=0.1 only mildly helps. Disagreement 18: 15 pick-wrong / 3 pick-right; argmax never recovers the true user among the 15.

## 2026-08-05 — Context-length gate + conditional Extract chunking (next)
We have server-side context limits for qwen3.5-4b (16384) and qwen3.6-35b-a3b-nvfp4 (32768). Next: compute REAL vLLM `usage.prompt_tokens` for the 10 longest T7/T8 Extract inputs, then implement Extract chunking only when the measured prompt exceeds 16384, and finally rerun the regression_50 baseline end-to-end to confirm no drift.

## 2026-08-05 — Fixed token-count script overflow handling
Updated `experiments/p1_context/01_token_counts_top10_t7_t8.py` to catch per-profile context-length errors, mark them as `exceeds_limit`, and continue so we still get real token counts for profiles that do fit under 16384.

## 2026-08-05 — Updated selection for boundary calibration
Revised the same script to select 10 profiles across `T5/T6/T7` whose word counts are closest to 16384, balancing just-below and just-above cases (instead of the 10 longest).

## 2026-08-05 — Bracket token audit (10k–15k words)
Measured real vLLM prompt tokens: template overhead **217** tokens; body ~**1.32–1.39 tokens/word**. Crossover: ~**12k words** (11,983 ok @ 16,237 tokens; 12,052 exceeds). Outputs: `results/token_counts_bracket_10k_15k.csv` + `_meta.json`.

## 2026-08-05 — Conditional Extract chunking implemented + tested

Gate = real vLLM `usage.prompt_tokens` (max_tokens=1 probe) with `CHUNK_GATE_TOKENS=14000` (margin under 16384). On exceed/overflow: split comment list, Extract each chunk, merge summaries. Recursive half-split fallback on context errors during Extract.

Code: `src/esrc/extract.py`, merge prompt `prompts/extract_merge.txt`, helpers `embed.py`/`search.py`, scripts `experiments/p1_baseline/01_test_extract_chunking.py` + `02_run_regression_50_baseline.py`.

**Chunking smoke** (`results/runs/p1_chunking_test_20260805T100126Z/`): 6/6 ok.
- under-gate (10,025w): single, probe=13452
- borderline/overflow/long: 2–8 chunks as expected

**regression_50 E→S→R** (`results/runs/p1_baseline_regression_50_20260805/`):
- Extract: 50/50 ok; **33 chunked / 17 single** (not re-run)
- Search: `search_top15.csv` complete (not re-run); Hit@15 **27/50** (fixture BSP gpt-4o ref 28/50)
- Reason: first pass 40/50 (1 JSON parse + 9 vLLM 500s); `--resume --phase reason` retried failures → **50/50 ok**
- Reason top-1: **24/50** (fixture ref 25/50)
- Single-path (n=17): top1 **10/17**, Hit@15 **12/17** — no failures on short profiles
- Chunked-path (n=33): top1 **14/33**, Hit@15 **15/33**
- `evaluate()` now prefers last ok per `query_user_id` so resume appends do not double-count

## 2026-08-05 — Reason concurrency bench (N=10, no full pool)

Script: `experiments/p1_baseline/03_reason_concurrency_bench.py`. Reused extract+search packs from `p1_baseline_regression_50_20260805` (no re-extract). Same 10 fixture users × K∈{1,2,4,8}; `enable_thinking=False`.

Artifacts: `results/runs/p1_reason_concurrency_bench_20260805T124648Z/`.

| K | wall_s | min/user | errors | proj h @500 | proj h @1000 |
|---|--------|----------|--------|-------------|--------------|
| 1 | 30.8 | 0.051 | 0/10 | 0.43 | 0.85 |
| 2 | 31.0 | 0.052 | 0/10 | 0.43 | 0.86 |
| 4 | 24.3 | 0.041 | 0/10 | 0.34 | 0.68 |
| 8 | 21.6 | 0.036 | 1/10 JSONDecode | 0.30 | 0.60 |

**Verdict:** K=4 is best stable (0% errors, ~1.27× vs sequential). K=2 ≈ no wall speedup (vLLM already keeps GPU busy). K=8 slightly faster but 1 parse error — not recommended for unattended runs. Sequential is ~3s/user with thinking off (≪ prior ~1.4 min/user estimate) → full ~1000 Reason is ~1h even at K=1; overnight unnecessary for Reason alone. Prefer K=4 + resume for pool_en / pool_en+HN.

## 2026-08-05 — Extract concurrency bench (N=10, 7 chunked + 3 single, no full pool)

Script: `experiments/p1_baseline/04_extract_concurrency_bench.py`. Intentional mix from `extract_meta.jsonl` (≈33:17 → 7+3). Same 10 users × K∈{1,2,4,8}; `qwen3.5-4b`, `enable_thinking=False`. Each concurrent user may issue probe/extract/merge calls.

Artifacts: `results/runs/p1_extract_concurrency_bench_20260805T125426Z/`.

| K | wall_s | errors | min/user | proj h @500 | proj h @1000 | notes |
|---|--------|--------|----------|-------------|--------------|-------|
| 1 | 587.4 | 0/10 | 0.979 | 8.16 | 16.32 | sequential baseline |
| 2 | 365.0 | 0/10 | 0.608 | 5.07 | 10.14 | **best wall** (~1.61×) |
| 4 | 382.9 | 0/10 | 0.638 | 5.32 | 10.64 | no gain vs K=2; latency inflation |
| 8 | 410.3 | 0/10 | 0.684 | 5.70 | 11.40 | worse wall; heavy contention |

Chunked vs single mean latency (ok users):

| K | chunked mean_s (n=7) | single mean_s (n=3) |
|---|----------------------|---------------------|
| 1 | 79.5 | 10.2 |
| 2 | 82.4 | 48.1 |
| 4 | 163.6 | 61.5 |
| 8 | 319.8 | 101.3 |

Mix66 projection (@1000, assume 66% chunked): K1 15.6h / K2 10.0h / K4 10.3h / K8 11.0h.

**Verdict:** All K stable (0% errors). Prefer **K=2** for Extract — best wall; K≥4 adds contention (chunked users issue multi-call storms) without throughput gain. Full pool_en Extract (~500–1000) ≈ **5–10h at K=2** → overnight preferred; working-session feasible for ~500 with resume. Reason remains cheap (~1h @K=4); Extract dominates E2E. No full pool run started.

## 2026-08-05 — Overnight pre-flight (pool_en + HN)

**Inputs:** pool_en T8 ready (500 queries, 1000 cand summaries, cand emb+index, prompts). HN ready under BSP layout (`data/truncated/T*`, `data/extracted_summaries/candidate`, mapping/manifest) — **no** `data/esrc/pool_hn`; wired via new runner.

**Fixes:**
- Added `experiments/p1_baseline/05_run_full_pool.py` (pool_en|hn|both, Extract K / Reason K, `--resume`, per-user jsonl+summary writes, subdirs per pool).
- Fixed extract `--resume` bug: `load_done_jsonl` required `query_user_id` but extract rows only had `user_id` → shared `src/esrc/resume.py`; baseline updated.
- HN query text from truncated **JSONL** (newline-separated comments); `data/summaries/T*` is space-joined and would break chunking.
- HN Search gallery: build/cache all-mpnet matrix from candidate summaries (BSP HN emb is OpenAI 1536-d).
- Created `summer_esrc/.env` (gitignored) for vLLM URL/key.

**Resume test:** extract `--limit 5`, kill after 2 ok → `--resume` logged `2 ok cached, 3 pending`, prior rows untouched. Reason: simulated error mid-file → resume retried failures only; evaluate uses last-ok-per-user.

**Disk:** 34Gi free; ~1000-user outputs ≪100MB → OK.

**Do not start full overnight from agent** — commands in chat for user (`caffeinate -i`). Expect ~5h extract pool_en @K=2 + ~0.5h reason; HN extract likely longer (mean ~33k words, almost all chunked).

## 2026-08-05 — results/ deliverable layout

```
results/
  runs/                 # raw run artifacts (untouched overnight dir)
  p1_baseline/{tables,figures}/
  p2_confidence/{tables,figures}/
  p3_calibration/{tables,figures}/
  p4_open_vs_frontier/{tables,figures}/
```

See `results/README.md`. Token-count CSVs moved into `p1_baseline/tables/`. Overnight `p1_full_pool_overnight_20260805/` left under `runs/`.

## 2026-08-05 — P2 (a)/(b) finalize + P3 isotonic (offline)

While overnight P1 runs (no extra server load):

**P2** (`experiments/p2_confidence/06_finalize_ab_stats.py` → `results/p2_confidence/tables/`):
- Metrics + reliability tables for (a)/(b)
- Bootstrap **10k** CIs on R@90/99%P, ECE, Brier, AP
- McNemar on thresholded correctness classifiers (τ=0.9, 0.99) + Bonferroni m=2
- Fixture: (a) R@90%P=0 CI[0.00,0.73]; (b) 0.78 CI[0.23,0.96] — still overlapping
- McNemar τ=0.9 n.s.; τ=0.99 significant (b better as τ=0.99 classifier), p≈0.001

**P3** (`experiments/p3_calibration/01_isotonic_fixture.py` → `results/p3_calibration/tables/`):
- Half-split seed=42; fit isotonic on cal, eval blind-0.9 vs isotonic-then-0.9 on test
- Logic OK: (b) ECE 0.32→0.20; naive@0.9 prec 0.80 → iso@0.9 prec 0.875 (n_test=25 noisy)
- (a) iso@0.9 accepts 0 on test (verbalized mass collapses) — expected at tiny n

**Estimator (c) drafts** (no server): `experiments/p2_confidence/DRAFT_estimator_c_prompts.md` — C1 binary YES/NO confirm-pick; C2 pairwise pick-vs-rival. Test after overnight.

## 2026-08-06 — Full pool done; Reason retry blocked; HN chunk audit

Overnight finished: pool_en Reason 445/500, HN 444/499 (55 errors each, mostly litellm 500 + JSON). Metrics not final.

**Reason retry:** First K=4 resume hung (model stopped answering). Extract still healthy. Background poller + sequential K=1 resume waiting for `qwen3.6-35b-a3b-nvfp4` to recover (likely drained by ~55 queued heavy prompts from the hung flood).

**HN merge-quality audit** (`results/runs/.../hn_chunk_audit/`): 5 users with 11–14 chunks vs gpt-4o archived summaries. 4/5 degraded (truncation mid-word, 60% dup loop on user_0961, trait dumps 5–10× longer than gpt-4o). user_0581 OK. Interim (incomplete denom) top1 by chunks on HN: 2→23%, 3–4→18%, 5–8→13%, 9+→10% — monotonic; Reddit shows no such decline.

## 2026-08-06 — Stuck 400 flood: diagnosis + client fix (do not resume yet)

IT: GPU pegged by one request repeating ~every 10s with **400 context exceeded** — `max_tokens=16000` + prompt **16769** on the **32k** Reason model (16769+16000>32768).

**Findings (local):**
- No `max_tokens=16000` anywhere in our repo; Reason/Extract runners default to **1024**.
- Our pending Reason prompts estimate ~3–5k tokens (max ~4.7k) — **not** 16.7k. Closest Extract probes are ~16.3k on `qwen3.5-4b`. Stuck payload is therefore not a normal Reason record-selection call from our jsonl; likely litellm default/other client, or a mis-routed Extract-sized prompt. Ask IT for request `model` + prompt fingerprint if still needed.
- No local `05_run_full_pool` / curl flooders running now (poller + K=4 retries already aborted). **Do not hit the server** until IT clears the queue.

**Client fixes shipped (no server calls):**
1. `get_client(max_retries=0)` — stop SDK blind resends.
2. `ContextLengthExceededError` / `PermanentRequestError` on context / HTTP 400; status=`permanent_error`.
3. `--resume` uses `load_resume_skip_user_ids` — skips ok **and** permanent failures (was re-queueing every error forever).
4. Per-model context map: Extract **16384**, Reason **32768**; `clamp_max_tokens` caps by max_output (8192) + remaining headroom.
5. Extract split-fallback still OK (smaller payload only); never resends the same overflowing request.

**Resume Reason retries only after IT confirms the repeating 400 is gone.**

## 2026-08-06 — P1 hand-in: bootstrap vs gpt-4o + HN chunk finding

Full denominators locked (pool_en 500, HN 499). Deliverables in `results/p1_baseline/`.

**C1 (Reddit ≈ gpt-4o):** top-1 36.2% vs 39.4%, Δ −3.2 pp, 95% CI includes 0 (McNemar n.s.). Hit@15 significantly worse (−9.8 pp).  
**C2 (HN transfer):** rejected — top-1 16.6% vs 38.0% (Δ −21.4 pp); Hit@15 23.8% vs 59.4% (Δ −35.5 pp); both CIs exclude 0.

**HN explanation:** accuracy and merge-quality proxies degrade monotonically with `n_chunks` (2–3→20% top-1 … 11+→0%); Reddit shows no such trend. Search collapse (Hit@15) is the main mechanism. See `HN_CHUNKING_FINDING.md`.

## 2026-08-06 — P2 full-pool (a)/(b) blocked: P1 preds pick-only

Overnight Reason jsonl has no `verbalized_confidence` / logprobs (lean P1 runner). Cannot compute ECE/Brier/R@P offline. Prepared:
- `experiments/p2_confidence/07_rescore_reason_ab_full_pool.py` — Reason-only re-score reusing P1 Extract+Search
- `experiments/p2_confidence/08_finalize_ab_full_pools.py` — metrics + bootstrap + McNemar Bonferroni k=4 + reliability
- Note: `results/p2_confidence/BLOCKED_full_pool_ab.md`

Estimator (c) still deferred until server stays stable; fixture prompt redesign first.
