# Bachelor's Seminar Paper
## Topic: How Much Text Is Enough? Measuring the Minimum Amount of Text Required for LLM-Based Deanonymization

**Student:** Hlib Petrov
**Supervisor:** Radu STATE
**Advisor:** Tatiana PETROVA
**Secondary Language:** French

---

## Overview and Research Question

A foundational result in privacy research (De Montjoye et al., 2013) showed that just four spatiotemporal data points are enough to uniquely identify 95% of individuals in mobile datasets. Your paper asks the text equivalent of this question: **how much text does a user need to produce online before they become deanonymizable by an LLM-based attacker?**

Lermen et al. (2026) contain one relevant observation: in their Reddit movie experiment, deanonymization recall increases with the number of movies discussed (3.1% recall for users discussing 1 movie vs. 48.1% for users discussing 10+). This is a single data point, not a systematic study. Your paper conducts that study.

You will measure deanonymization success (recall) across 8 levels of text volume, find the minimum threshold at which deanonymization becomes possible, and investigate two additional dimensions: whether content **diversity** matters more than raw volume, and whether different **types of content** (personal vs. opinion vs. topical) differ in their identifying power.

---

## Mandatory Reading — Complete Before Any Experiments

1. **Main paper:** Lermen et al. (2026), *"Large-scale online deanonymization with LLMs"*, arXiv:2602.16800
   - Read fully. Pay special attention to: Section 5 (especially Figure 4b — your closest existing result), Section 6, Appendix G (full pipeline and prompts).
   - Available at: https://arxiv.org/abs/2602.16800

2. **Foundational analogy:** De Montjoye et al. (2013), *"Unique in the Crowd: The Privacy Bounds of Human Mobility"*
   - https://www.nature.com/articles/srep01376
   - This is the conceptual inspiration for your research question. Read to understand how a "minimum sufficient data" result is formulated and reported.

3. **On changepoint detection:** `ruptures` library documentation
   - https://centre-borelli.github.io/ruptures-docs/
   - You will use this library to automatically detect the point on your recall curve where performance stops improving significantly.

---

## Tools and Environment Setup

### Step 0.1 — Install Python and core tools

- Python 3.10 or higher: https://www.python.org/downloads/
- Create a virtual environment: `python -m venv venv`, then activate it
- Jupyter Notebook: https://jupyter.org/install

### Step 0.2 — Install required Python libraries

| Library | Purpose | Documentation |
|---------|---------|---------------|
| `pandas` | Working with tabular data | https://pandas.pydata.org |
| `numpy` | Numerical operations | https://numpy.org |
| `zstandard` | Reading compressed Reddit dumps (.zst files) | https://github.com/indygreg/python-zstandard |
| `jsonlines` | Reading Reddit dumps line by line | https://jsonlines.readthedocs.io |
| `sentence-transformers` | Computing text embeddings (free, local) | https://www.sbert.net |
| `faiss-cpu` | Fast nearest-neighbor search | https://github.com/facebookresearch/faiss |
| `openai` | Access to GPT via API | https://platform.openai.com/docs |
| `tqdm` | Progress bars | https://tqdm.github.io |
| `scikit-learn` | Precision and recall metrics | https://scikit-learn.org |
| `matplotlib` | Plotting | https://matplotlib.org |
| `scipy` | Statistical tests (Spearman correlation) | https://scipy.org |
| `ruptures` | Changepoint detection on the recall curve | https://centre-borelli.github.io/ruptures-docs/ |
| `google-cloud-bigquery` | Accessing Hacker News data via BigQuery | https://cloud.google.com/bigquery/docs/reference/libraries |

### Step 0.3 — Set up your OpenAI API account

- Register at: https://platform.openai.com
- Create an API key under "API keys"
- Ask your advisor for a budget allocation
- **Security rule:** Store your key in a `.env` file — never write it in code, never upload to GitHub

**Model selection and cost guidelines:**

| Step | Recommended model | Why |
|------|------------------|-----|
| Extract — all truncation levels | `gpt-4o-mini` | Simple summarization — cheap and sufficient |
| Content type classification | `gpt-4o-mini` | Simple classification task |
| Reason — development and testing | `gpt-4o-mini` | Use for ALL runs during development and debugging |
| Reason — final run only | `gpt-4o` | Switch to the stronger model only for the final reported results |

**Why this matters — especially for this paper:** This paper runs the ESRC pipeline 8 times (one per truncation level) plus additional runs for diversity and content type experiments. With `gpt-4o` for all Reason steps, total cost reaches **$100–$180**. Using `gpt-4o-mini` throughout development and switching to `gpt-4o` only for the final run reduces cost to approximately **$20–$30** during development, with a one-time final cost of **$50–$80**.

Since you are measuring *differences across truncation levels*, not absolute recall values, comparisons remain scientifically valid even when using `gpt-4o-mini`. Use `gpt-4o` only for the single final run that produces your reported numbers.

**Estimated total API cost for this paper: $100–$180** (final run with `gpt-4o`). This is the most expensive of the four topics — plan and track your budget carefully.

### Step 0.4 — Set up Google BigQuery

- Go to: https://console.cloud.google.com
- Create a new Google Cloud project
- Enable the BigQuery API
- Free tier: 1 TB of queries per month — more than sufficient for this project
- Quickstart guide: https://cloud.google.com/bigquery/docs/quickstarts/query-public-dataset-console

---

## Understanding the ESRC Framework

**E — Extract:** Feed a user's comments to an LLM. Ask it to summarize the profile using the exact Summarization Prompt from Appendix G.2 of Lermen et al. (2026). Result: a structured list of attributes (location, profession, interests, values, demographics).

**S — Search:** Embed all profile summaries using `sentence-transformers` (`all-mpnet-base-v2`). Build a FAISS index over candidate profiles. For each query, retrieve the top-15 most similar candidates.

**R — Reason:** Feed query summary + top-15 candidates to an LLM. Ask it to select the best match and give a confidence score (0–1). Use `gpt-4o-mini` during development; `gpt-4o` for the final run only.

**C — Calibrate:** Vary the confidence threshold to build a Precision-Recall curve.

---

## Key Metrics

**Precision:** `TP / (TP + FP)` — fraction of guesses that are correct.

**Recall:** `TP / (TP + FN)` — fraction of true matches found.

**Recall@90% Precision:** Maximum recall while keeping precision ≥ 90%. Primary metric.

**Recall@99% Precision:** Same, stricter threshold.

**Minimum Sufficient Text (MST) — your key metric:**
The smallest truncation level T at which Recall@90% Precision exceeds 5%. This is your operationalized definition of "deanonymizability". Values below 5% are treated as noise.

**Spearman correlation:** Measure the monotonic relationship between text volume and recall. Use `scipy.stats.spearmanr`. A value close to 1 confirms that more text consistently leads to higher recall.

**How to build a Precision-Recall curve:** Sort all predictions by descending confidence. At each threshold, compute precision and recall. Plot: X-axis = recall, Y-axis = precision.

---

## Datasets

### Primary Dataset: Pushshift Reddit Comments Dumps

- Source: Academic Torrents
- URL: https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4
- Alternative: https://the-eye.eu/redarcs/

**What to download:**
You need a 4-year span to enable a robust temporal split with enough data on each side. Download:
- From 2015–2016 (query side): `RC_2015-06.zst`, `RC_2016-01.zst`, `RC_2016-06.zst`, `RC_2017-01.zst`
- From 2019–2020 (candidate side): `RC_2019-06.zst`, `RC_2020-01.zst`

Each file is 5–15 GB compressed. Read using `zstandard` line by line — never load full files into memory.

**Format:** Each line = one JSON with `author`, `body`, `subreddit`, `created_utc`, `score`.

**This dataset requires stricter activity filters than in other topics** — see Step 1.

### Secondary Dataset: Hacker News via Google BigQuery

- Access: https://console.cloud.google.com/bigquery
- Tables: `bigquery-public-data.hacker_news.comments` and `bigquery-public-data.hacker_news.stories`
- Free access, no download needed — query directly in the browser or via Python

Hacker News is a technical professional community with a very different writing style from Reddit. If your minimum-text threshold holds for both platforms, it is more universally valid. If it differs, that is also a scientific finding.

---

## Step-by-Step Experimental Instructions

---

### Step 1 — Filter Reddit Users (Stricter Criteria)

**Why stricter than other papers:** You need to truncate profiles to as few as 5 comments. If a user only has 200 comments total, you cannot study behaviour across all 8 truncation levels — you run out of data. You therefore need users with substantially more activity.

Apply these filters:

| Criterion | Value | Reason |
|-----------|-------|--------|
| Minimum comments **per split side** | ≥ 500 | Enough to study all truncation levels including T7 (500 comments) |
| Activity span | ≥ 4 years | Ensures a valid 1-year gap between query and candidate |
| Max average comments per day | ≤ 24 | Exclude bots |
| Username | Does not end in `bot`, `gpt`, `mod` | Exclude bots |
| Account | Not `[deleted]` | Unusable |

Process: Read all dump files, compute per-author statistics (comment count, date range), apply filters, save the list of qualifying usernames.

**Target size after filtering:** 1,000–2,000 users.

---

### Step 2 — Create Temporal Splits

For each qualifying user, split their comment history into two non-overlapping profiles.

**Splitting procedure:**
- Find T* = the midpoint of the user's activity (by comment count)
- **Query profile** = all comments before (T* − 182 days)
- **Candidate profile** = all comments after (T* + 182 days)
- Comments within 365 days of T* are **discarded**

**Verify the split:** Count comments on each side. You need ≥ 500 on each side. Discard users below this threshold after splitting.

**Sort comments chronologically** within each profile — this is critical for the truncation step.

Save as:
- `user_XXXXX_query_full.jsonl` — all query-side comments, in chronological order
- `user_XXXXX_candidate.jsonl` — all candidate-side comments (used as-is)

---

### Step 3 — Create 8 Truncation Levels

For each user, create 8 truncated versions of the query profile by taking the **first N comments in chronological order**:

| Condition | Comments | Approximate word count (average) |
|-----------|----------|----------------------------------|
| T1 | 5 | ~200–300 words |
| T2 | 10 | ~400–600 words |
| T3 | 25 | ~1,000–1,500 words |
| T4 | 50 | ~2,000–3,000 words |
| T5 | 100 | ~4,000–6,000 words |
| T6 | 200 | ~8,000–12,000 words |
| T7 | 500 | ~20,000–30,000 words |
| T8 | Full profile | All available comments |

**Always take the first N chronologically** — this simulates an attacker who only has access to a user's early activity history.

**Count the actual word count** for each condition, averaged across all users. Report these numbers in your paper alongside comment counts.

Save all 8 versions per user: `user_XXXXX_query_T1.jsonl`, `user_XXXXX_query_T2.jsonl`, ..., `user_XXXXX_query_T8.jsonl`.

---

### Step 4 — Collect Hacker News Data

In the Google BigQuery web console (https://console.cloud.google.com/bigquery), write and run a SQL query to:
1. Count comments per author from `bigquery-public-data.hacker_news.comments`
2. Filter authors with ≥ 500 total comments and ≥ 4 years of activity (from `MIN(time)` to `MAX(time)`)
3. Exclude `[deleted]` and bot-like usernames

Then run a second query to download all comments for the qualifying authors.

**Temporal split for HN:** Apply the same procedure as for Reddit (Step 2). The comment field in BigQuery is `text`, and the timestamp field is `time` (Unix timestamp).

**Apply the same 8 truncation levels** as for Reddit (Step 3).

**Target size:** 500–1,000 users.

---

### Step 5 — Run ESRC Across All Truncation Levels (Reddit)

**Step 5.1 — Extract**

For each user, for each truncation level T1–T8: feed the truncated comment set to the OpenAI API using the exact Summarization Prompt from Appendix G.2 of Lermen et al. (2026).

- Model: `gpt-4o-mini`
- Save all summaries to disk immediately

**Important cost note:** This means you call the Extract API 8 times per user. For 1,000 users, that is 8,000 API calls. Test on 100 users first. Extract is cheap (`gpt-4o-mini`) — the main cost comes from the Reason step.

**Step 5.2 — Search**

Build a single FAISS index from the candidate profile embeddings — this index is the same for all 8 truncation levels.

For each truncation level, embed the corresponding query summaries and retrieve top-15 candidates per query.

**Step 5.3 — Reason**

**During development:** Use `gpt-4o-mini`. Run on a sample of 200 users to verify your pipeline works correctly and produces meaningful results across truncation levels.

**For the final reported run:** Switch to `gpt-4o`. Run on the full dataset. This is the expensive step — for 1,000 users × 8 truncation levels, expect approximately $56 in Reason costs alone.

Save: predicted candidate ID + confidence score for each query, for each truncation level.

**Step 5.4 — Calibrate**

For each truncation level, build a Precision-Recall curve and extract Recall@90% and Recall@99% Precision.

You now have 8 values of Recall@90% (one per truncation level). This is your main result.

---

### Step 6 — Changepoint Detection

You have a sequence of recall values: Recall(T1), Recall(T2), ..., Recall(T8).

Plot this as a curve: X-axis = number of comments (5, 10, 25, 50, 100, 200, 500, full), Y-axis = Recall@90% Precision.

Use the `ruptures` library to detect the changepoint — the point at which the recall curve's slope changes significantly.

- Algorithm to use: `Pelt` with model `"rbf"` (radial basis function)
- Documentation: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/

The changepoint is your **Minimum Sufficient Text (MST)**: the text volume beyond which additional content provides diminishing returns for the attacker.

Also define a simpler operational MST: the smallest T where Recall@90% > 5%.

Report both definitions and compare them.

---

### Step 7 — Additional Experiment: Volume vs. Diversity

This is the second scientific contribution of your paper.

**Research question:** At the same text volume (T4 = 50 comments), does a user become more identifiable if their comments span many different communities (high diversity) versus being concentrated in one or two communities (low diversity)?

**Procedure:**

For each user at truncation level T4, count the number of **unique subreddits** in their first 50 comments. Divide users into three groups:

| Group | Unique subreddits in first 50 comments |
|-------|----------------------------------------|
| Low diversity | 1–3 subreddits |
| Medium diversity | 4–10 subreddits |
| High diversity | 11+ subreddits |

Run the full ESRC pipeline (using T4 query profiles) separately for each group.

Report Recall@90% for each group. If the high-diversity group has significantly higher recall → community diversity is more identifying than raw text volume.

**Model for this experiment:** Use `gpt-4o-mini` for Reason — you are comparing groups, not reporting absolute values, so the model choice does not affect the validity of the comparison.

---

### Step 8 — Additional Experiment: Content Type

This is the third scientific contribution of your paper.

**Research question:** Are some types of comments more identifying than others?

- **Personal (P):** comments that reveal information about the user's life ("my wife", "when I was in college", "I work as")
- **Opinion (O):** comments expressing values or beliefs ("I think", "I believe", "in my opinion")
- **Topical (T):** comments about a topic without personal disclosure ("this algorithm works by...", "the movie was great")

**Step 8.1 — Classify comments by type**

For each comment in the T4 profiles, call the OpenAI API (`gpt-4o-mini`) with this instruction:

*"Classify the following Reddit comment into exactly one category. P = personal disclosure (the author reveals facts about their own life). O = opinion or value statement (the author expresses a belief, preference, or judgment). T = topical discussion (the author discusses a topic without personal disclosure). Respond with only the letter P, O, or T."*

Save the classification for each comment.

**Step 8.2 — Create type-filtered profiles**

For each user, create three additional T4 profiles:
- P-only: first 50 comments of type P
- O-only: first 50 comments of type O
- T-only: first 50 comments of type T

Only include users who have ≥ 50 comments of each type (this will reduce sample size — report how many users qualify).

**Step 8.3 — Run ESRC on type-filtered profiles**

Use `gpt-4o-mini` for Reason in this experiment. You are comparing types against each other, not reporting absolute values.

---

### Step 9 — Replicate on Hacker News

Repeat Steps 5 and 6 (main truncation experiment) for the Hacker News dataset.

Use `gpt-4o-mini` for the development run, `gpt-4o` for the final reported run.

Compare the Minimum Sufficient Text threshold between Reddit and Hacker News.

---

### Step 10 — Compute Spearman Correlation

Using `scipy.stats.spearmanr`, compute the Spearman rank correlation between:
- X = number of comments (5, 10, 25, 50, 100, 200, 500, full)
- Y = Recall@90% Precision

A value close to 1 (p < 0.05) confirms the monotonic relationship between text volume and deanonymizability.

---

### Step 11 — Translate Results into Practical Terms

Your MST is currently expressed as "number of comments". Translate it into more interpretable terms:

1. **Word count:** Use the average word counts per truncation level that you computed in Step 3.
2. **Time of activity:** Compute the average time needed to accumulate T_MST comments in your dataset (e.g., "the average Reddit user in our dataset posts X comments per month, so T_MST corresponds to approximately Z months of activity").
3. **Number of subreddits covered:** Compute the average number of unique subreddits in T_MST comments.

---

## Results to Report

**Quantitative results:**
- Table: Recall@90% and Recall@99% for T1–T8 (Reddit)
- Table: Recall@90% and Recall@99% for T1–T8 (Hacker News)
- Figure: Recall@90% as a function of comment count, with changepoint marked (both platforms on one plot)
- MST (both definitions) for Reddit and Hacker News
- Spearman correlation coefficient
- Table: Recall@90% for low/medium/high diversity groups at T4
- Table: Recall@90% for P-only / O-only / T-only profiles at T4

**Qualitative results:**
- Practical translation of MST into months of activity and approximate word counts
- Discussion of which type of content is most identifying

---

## Final Report Structure

1. **Introduction** — analogy to De Montjoye et al. (2013); research question; practical significance
2. **Related Work** — Lermen et al. (2026) (Figure 4b as motivation), De Montjoye et al. (2013), Sweeney (2002) on k-anonymity
3. **Methodology** — datasets (Reddit high-activity filter, HN via BigQuery), temporal split, 8 truncation levels, diversity experiment, content type experiment, changepoint detection
4. **Results** — recall curves, MST, Spearman correlation, diversity results, content type results, HN comparison
5. **Discussion** — what does the MST mean in practice? Does diversity matter more than volume? Which type of content is most dangerous to share?
6. **Conclusion** — concrete recommendations for users: how many posts is "too many"?

---

## Practical Guidelines

### Budget Management

**Estimated cost breakdown:**

| Step | Model | Cost (1,000 Reddit + 500 HN users) |
|------|-------|-------------------------------------|
| Extract × 8 truncation levels — Reddit (1,000 users) | gpt-4o-mini | ~$18.00 |
| Extract — candidate profiles (1,000 users) | gpt-4o-mini | ~$1.00 |
| Extract × 8 truncation levels — HN (500 users) | gpt-4o-mini | ~$9.00 |
| Content type classification (T4 profiles) | gpt-4o-mini | ~$2.00 |
| Reason × 8 levels × 1,000 users — Reddit (final run) | gpt-4o | ~$56.00 |
| Reason × 8 levels × 500 users — HN (final run) | gpt-4o | ~$28.00 |
| Reason — diversity and content type experiments | gpt-4o-mini | ~$13.00 |
| Testing and debugging | gpt-4o-mini | ~$15.00 |
| **Total** | | **~$142** |

**Key cost-saving rule — critical for this paper:** This is the most expensive of the four topics because the ESRC pipeline runs 8 times. Use `gpt-4o-mini` for the Reason step during **all development and debugging**. Switch to `gpt-4o` only for the single final run that produces your reported numbers. This reduces development cost by 15–20×.

Your comparisons across truncation levels remain valid regardless of which model you use — as long as you use the **same model consistently** for all 8 truncation levels in the same run.

**Additional cost rules:**
- Run T1 and T8 first. If they show the expected pattern (low recall at T1, high at T8), proceed with T2–T7
- Save all Extract outputs immediately after generation — they are reused for all 8 Reason runs
- Keep a weekly log of API spending
- Test on 100 users before scaling to the full dataset

### Reproducibility
- Set `random_seed = 42` for all random sampling
- Log all model parameters: model name, temperature, max tokens
- Store code in Git with descriptive commit messages

### Ethics
- Do not publish individual user data
- Use only anonymous IDs (user_0001, user_0002, etc.)
- Confirm ethics requirements with your supervisor before data collection

### Recommended Folder Structure
```
project/
├── data/
│   ├── raw/                   ← original dump files
│   ├── filtered/              ← qualifying user lists (Reddit, HN)
│   ├── splits/
│   │   ├── query_full/        ← full query profiles (chronologically sorted)
│   │   └── candidate/         ← candidate profiles
│   ├── truncated/
│   │   ├── T1/ T2/ ... T8/    ← one folder per truncation level
│   ├── diversity_groups/      ← low/medium/high diversity subsets
│   ├── content_types/         ← P-only / O-only / T-only subsets
│   └── summaries/             ← Extract outputs (one folder per truncation level)
├── experiments/
│   ├── truncation/
│   ├── diversity/
│   └── content_type/
├── results/
│   ├── tables/
│   └── figures/
├── notebooks/
└── README.md
```

### Timeline (16 weeks)

| Weeks | Task |
|-------|------|
| 1 | Read all papers, set up environment, set up BigQuery |
| 2–3 | Download Reddit dumps, filter users, create temporal splits, collect HN data, create all 8 truncation levels |
| 4–5 | Run ESRC on T1–T8 for Reddit — development run (gpt-4o-mini), then final run (gpt-4o) |
| 6 | Run ESRC on T1–T8 for Hacker News (final run), changepoint detection |
| 7 | Diversity experiment, content type classification and experiment |
| 8–9 | Analysis, practical translation of MST, tables, figures |
| 10–12 | Write report, final edits, submission — **weeks 10–12 also serve as buffer** |

---

## Bibliography

All works referenced in this document. Read them in the order listed.

**[1] Lermen, S., Paleka, D., Swanson, J., Aerni, M., Carlini, N., & Tramèr, F. (2026).**
*Large-scale online deanonymization with LLMs.*
arXiv:2602.16800.
https://arxiv.org/abs/2602.16800
→ **Main paper. Read fully before starting any experiment. Figure 4b is your closest existing result.**

**[2] De Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D. (2013).**
*Unique in the crowd: The privacy bounds of human mobility.*
Scientific Reports, 3, 1376.
https://www.nature.com/articles/srep01376
→ **Conceptual inspiration for your research question. Shows how few data points suffice to identify individuals.**

**[3] Sweeney, L. (2002).**
*k-Anonymity: A model for protecting privacy.*
International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5), 557–570.
https://doi.org/10.1142/S0218488502001648
Find via Google Scholar: https://scholar.google.com (search "Sweeney 2002 k-anonymity")
→ **Classic privacy framework referenced in the Related Work section.**

**[4] Truong, C., Oudre, L., & Vayatis, N. (2020).**
*Selective review of offline change point detection methods.*
Signal Processing, 167, 107299.
https://arxiv.org/abs/1801.00718
`ruptures` library: https://centre-borelli.github.io/ruptures-docs/
→ **The changepoint detection method you use to find the Minimum Sufficient Text threshold (Step 6).**
