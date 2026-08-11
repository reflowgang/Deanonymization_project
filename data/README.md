# Data

This folder follows the **recommended layout** from `topic3 (1).md` (supervisor assignment).

| Path | Purpose |
|------|---------|
| `raw/` | Original Reddit `.zst` dumps (and optional HN exports). |
| `filtered/` | Lists of qualifying usernames / user IDs (Reddit, HN). |
| `splits/query_full/` | Full query-side profiles, chronological. |
| `splits/candidate/` | Candidate-side profiles. |
| `truncated/T1/` … `T8/` | Truncated query profiles per level. |
| `diversity_groups/` | Low / medium / high diversity subsets (T4). |
| `content_types/` | P-only / O-only / T-only profile subsets. |
| `summaries/T1/` … `T8/` | LLM Extract outputs per truncation level. |

Legacy test file: `comments_synthetic.jsonl` may stay in `data/` or move under `raw/` if you prefer.

---

Place processed datasets here according to the table above.

## Expected format

The pipeline expects a **JSON Lines** (`.jsonl`) file where each line is a JSON object with at least:

- **`user_id`** (str): Unique identifier for the user.
- **`text`** (str): Content of one comment (or post).
- **`timestamp`** (str or number): Ordering for temporal split (e.g. ISO date or Unix time). If missing, order of lines per user can be used.

### Example

```json
{"user_id": "u_001", "text": "I really enjoyed that movie.", "timestamp": "2023-01-15T10:00:00Z"}
{"user_id": "u_001", "text": "Same here, the ending was great.", "timestamp": "2023-02-20T14:30:00Z"}
{"user_id": "u_002", "text": "Has anyone tried the new recipe?", "timestamp": "2023-01-10T09:00:00Z"}
```

## Config

Set the path to this file in `config/default.yaml` under `data.input_path` (e.g. `data/comments.jsonl`).

## Synthetic data

For testing without real data, you can generate a small synthetic dataset (e.g. random “user” IDs and placeholder text) and point the config to it.
