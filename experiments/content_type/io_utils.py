from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from config import RANDOM_SEED, paths

def load_query_user_ids(manifest_csv: Path | None=None) -> list[str]:
    manifest_path = manifest_csv or paths()['manifest']
    df = pd.read_csv(manifest_path)
    return df.loc[df['has_query'] == 1, 'user_id'].astype(str).sort_values().tolist()

def select_pilot_users(user_ids: list[str], n_users: int, seed: int=RANDOM_SEED) -> list[str]:
    if n_users >= len(user_ids):
        return user_ids
    series = pd.Series(user_ids)
    return series.sample(n=n_users, random_state=seed).sort_values().tolist()

def iter_sorted_comments(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, str]] = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for (line_no, line) in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t_val = int(obj['t'])
            body = str(obj.get('b', '')).strip()
            if body:
                rows.append((t_val, line_no, body))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [{'comment_index': i + 1, 'timestamp': t_val, 'source_line': line_no, 'body': body} for (i, (t_val, line_no, body)) in enumerate(rows)]

def pilot_paths() -> dict[str, Path]:
    root = paths()['manifest'].parent
    data_root = paths()['type_profiles_root'].parent / 'content_types_pilot'
    return {'classifications_csv': root / 'content_type_pilot_classifications.csv', 'classify_log_csv': root / 'content_type_pilot_classify_log.csv', 'qualification_csv': root / 'content_type_pilot_qualification.csv', 'summary_csv': root / 'content_type_pilot_summary.csv', 'type_profiles_root': data_root}

def production_paths() -> dict[str, Path]:
    p = paths()
    return {'classifications_csv': p['classifications_csv'], 'classify_log_csv': p['classify_log_csv'], 'qualification_csv': p['qualification_csv'], 'summary_csv': p['summary_out'], 'type_profiles_root': p['type_profiles_root']}

def resolve_run_paths(pilot: bool) -> dict[str, Path]:
    return pilot_paths() if pilot else production_paths()
