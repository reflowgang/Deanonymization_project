# Estimator (c) — redesigned forced-hypothesis prompts

**Status:** implementation ready — `experiments/p2_confidence/09_c_redesign_c1_c2.py`
(prompts under `prompts/estimator_c1_confirm.txt`, `prompts/estimator_c2_pairwise.txt`).
Server smoke blocked 2026-08-06 (gpu6 connection timeout); re-run when vLLM is up.

**Problem with current (c):** forced logprob over “is candidate *k* the match?”
yields near-uniform softmax mass (~1/15). Argmax ≠ pick on 18/50; never
recovers the true user among pick-wrong cases. Tiny raw lp spreads → C3-looking
signal that may be a **prompt artifact**.

Goal: force the model to *compare against the already-chosen pick* (or commit
to a binary decision) so the score for the selected id is not washed out by
15 near-identical completions.

---

## Variant C1 — “Confirm / reject the pick” (binary)

Use after the free Reason pick is known (`selected = k`).

```
You previously selected candidate {k} as the best match for the anonymous query profile.

Query profile:
{query_summary}

Selected candidate {k}:
{candidate_k_summary}

Task: Confirm or reject this selection.
Answer with exactly one token: YES or NO
- YES = candidate {k} is the same person as the query author
- NO  = candidate {k} is not the same person
```

**Score:** `score_c = P(YES) = exp(logprob_YES) / (exp(logprob_YES)+exp(logprob_NO))`
(or just sigmoid of logprob_YES − logprob_NO).

**Why it might help:** binary forced choice; mass can’t dilute across 15 options.
**Risk:** still verbalizes agreement with its own pick → may stay overconfident YES.

---

## Variant C2 — “Pick vs each rival” (pairwise margin)

For each rival `j ≠ k` (or top-3 rivals by Search score):

```
Anonymous query profile:
{query_summary}

Candidate A (#{k}):
{candidate_k_summary}

Candidate B (#{j}):
{candidate_j_summary}

Which candidate is the same person as the query author?
Answer with exactly one token: A or B
```

**Score:** geometric mean (or min) of `P(A)` over rivals; or
`score_c = softmax_temperature( { log P(A vs j) }_j )` aggregated.

Simpler deliverable score: `score_c = mean_j P(A beats j)`.

**Why it might help:** relative evidence vs alternatives; margins should shrink
when the pick is weak.
**Risk:** 14× (or 3×) extra calls per query — defer full 15 until a 10-user
smoke after overnight; start with top-3 rivals only.

---

## What we will *not* change yet

- Softmax-over-15 independent “is this the one?” prompts (current design).
- Temperature hacks alone (T=0.1 only mildly helped).

## Test plan (post-overnight)

1. Smoke n=10 fixture users, variants C1 and C2-top3, no thinking.
2. Check: score_c mean/spread, correlation with correct, R@90%P, argmax-vs-pick N/A for C1.
3. If C1 still flat-high YES → evidence for C3 (no separable evidence) rather than prompt artifact.
4. If C2 spreads and ranks better → adopt as bonus estimator (c′).

## Open design questions (for discussion)

- C1: allow `YES`/`NO` only vs short JSON `{"confirm": true}`?
- C2: top-3 rivals by FAISS vs all 14?
- Should (c) be defined only when Reason pick is available (yes — forced-hypothesis on the pick)?
