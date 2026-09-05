from __future__ import annotations
import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
MODEL = 'gpt-4o'
TEMPERATURE = 0
SLEEP_SECONDS = 0.2
PROMPT_REL_PATH = Path('prompts/record_selection_lermen_g2.txt')
LEVELS_ALL: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']

@dataclass(frozen=True)
class Paths:
    faiss_csv: Path
    query_summaries_root: Path
    candidate_summaries_dir: Path
    out_csv: Path
    prompt_file: Path

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_args(argv: Optional[list[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Reason over FAISS top-15 using the Lermen G.2 record-selection prompt.')
    p.add_argument('--levels', nargs='+', default=LEVELS_ALL, help='Levels to process, e.g. --levels T1 T2 (default: all).')
    p.add_argument('--limit', type=int, default=None, help='Limit number of query users per level (for testing).')
    return p.parse_args(argv)

def normalize_levels(levels: list[str]) -> list[str]:
    levels_norm = [lvl.strip() for lvl in levels if lvl and lvl.strip()]
    unknown = sorted(set(levels_norm) - set(LEVELS_ALL))
    if unknown:
        raise ValueError(f'Unknown levels: {unknown}. Allowed: {LEVELS_ALL}')
    out: list[str] = []
    seen: set[str] = set()
    for lvl in levels_norm:
        if lvl not in seen:
            out.append(lvl)
            seen.add(lvl)
    return out

def load_api_key_from_env(project_root: Path) -> None:
    env_path = project_root / 'api_key.env'
    if not env_path.exists():
        env_path = project_root / '.env'
    load_dotenv(env_path, override=False)
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError(f'OPENAI_API_KEY not found. Expected it in {env_path}.')

def get_client(project_root: Path) -> OpenAI:
    load_api_key_from_env(project_root)
    return OpenAI()

def load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f'Prompt file not found: {prompt_path}')
    return prompt_path.read_text(encoding='utf-8')

def build_user_prompt(template: str, query_summary: str, candidate_block: str) -> str:
    prompt = template
    if '{query_summary}' in prompt:
        prompt = prompt.replace('{query_summary}', query_summary)
    else:
        prompt = prompt.rstrip() + '\n\nQUERY:\n' + query_summary
    if '{candidate_block}' in prompt:
        prompt = prompt.replace('{candidate_block}', candidate_block)
    else:
        prompt = prompt.rstrip() + '\n\nCANDIDATES:\n' + candidate_block
    return prompt

def build_candidate_block(candidates: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for (i, c) in enumerate(candidates, start=1):
        cid = str(c['candidate_user_id'])
        rank = int(c['rank'])
        score = float(c['score'])
        summary = str(c['summary'])
        parts.append(f'[{i}] candidate_user_id: {cid}\n    rank: {rank}\n    score: {score:.6f}\n    summary: {summary}\n')
    return '\n'.join(parts).strip()

def parse_candidate_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.match('^(?:candidate\\s*)?(\\d+)$', s, re.I)
    if m:
        return int(m.group(1))
    m = re.match('^user_(\\d+)$', s, re.I)
    if m:
        return int(m.group(1))
    return None

def resolve_predicted_candidate_id(out_obj: dict[str, Any], candidate_ids: list[str]) -> tuple[str, str]:
    n = len(candidate_ids)
    if n == 0:
        return ('', 'no candidates provided')
    num = parse_candidate_number(out_obj.get('selected_candidate_number'))
    if num is not None:
        if 1 <= num <= n:
            return (candidate_ids[num - 1], '')
        return ('', f'selected_candidate_number out of range: {num} (valid 1..{n})')
    for key in ('predicted_candidate_id', 'predicted_candidate_user_id'):
        raw_id = str(out_obj.get(key, '')).strip()
        if not raw_id:
            continue
        if raw_id in candidate_ids:
            return (raw_id, '')
        as_num = parse_candidate_number(raw_id)
        if as_num is not None and 1 <= as_num <= n:
            return (candidate_ids[as_num - 1], '')
        return ('', f'could not map prediction to candidate list: {raw_id!r}')
    return ('', 'missing selected_candidate_number and predicted_candidate_id')

def read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return obj

def summary_obj_to_text(summary_obj: dict[str, Any]) -> str:
    summary = summary_obj.get('summary')
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return json.dumps(summary_obj, ensure_ascii=False)

def load_existing_predictions(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    df = pd.read_csv(out_csv)
    if df.empty:
        return set()
    if 'level' not in df.columns or 'query_user_id' not in df.columns:
        return set()
    return {(str(r['level']), str(r['query_user_id'])) for (_, r) in df.iterrows()}

def ensure_out_csv_header(out_csv: Path) -> None:
    _ensure_parent_dir(out_csv)
    if out_csv.exists():
        return
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['level', 'query_user_id', 'predicted_candidate_id', 'true_candidate_id', 'confidence', 'is_correct', 'status', 'error'])
        w.writeheader()

def append_row(out_csv: Path, row: dict[str, Any]) -> None:
    with out_csv.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['level', 'query_user_id', 'predicted_candidate_id', 'true_candidate_id', 'confidence', 'is_correct', 'status', 'error'])
        w.writerow(row)

def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def parse_model_json(content: str) -> dict[str, Any]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f'Model returned invalid JSON: {e}') from e
    if not isinstance(obj, dict):
        raise ValueError('Model returned non-object JSON.')
    return obj

def call_openai_reason(client: OpenAI, prompt_template: str, query_summary: str, candidate_block: str) -> dict[str, Any]:
    user_content = build_user_prompt(prompt_template, query_summary, candidate_block)
    resp = client.chat.completions.create(model=MODEL, temperature=TEMPERATURE, messages=[{'role': 'user', 'content': user_content}], response_format={'type': 'json_object'})
    content = resp.choices[0].message.content or ''
    return parse_model_json(content)

def main() -> None:
    args = parse_args()
    levels = normalize_levels(list(args.levels))
    limit = args.limit
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(faiss_csv=project_root / 'results/tables/hn_faiss_top15.csv', query_summaries_root=project_root / 'data/extracted_summaries', candidate_summaries_dir=project_root / 'data/extracted_summaries/candidate', out_csv=project_root / 'results/tables/hn_reason_predictions.csv', prompt_file=project_root / PROMPT_REL_PATH)
    if not paths.faiss_csv.exists():
        raise FileNotFoundError(f'FAISS results CSV not found: {paths.faiss_csv}')
    prompt_template = load_prompt_template(paths.prompt_file)
    ensure_out_csv_header(paths.out_csv)
    done = load_existing_predictions(paths.out_csv)
    faiss_df = pd.read_csv(paths.faiss_csv)
    required_cols = {'level', 'query_user_id', 'rank', 'candidate_user_id', 'similarity'}
    missing = required_cols - set(faiss_df.columns)
    if missing:
        raise ValueError(f'FAISS CSV missing columns: {sorted(missing)}')
    faiss_df['level'] = faiss_df['level'].astype(str)
    faiss_df['query_user_id'] = faiss_df['query_user_id'].astype(str)
    faiss_df['candidate_user_id'] = faiss_df['candidate_user_id'].astype(str)
    faiss_df['rank'] = faiss_df['rank'].astype(int)
    faiss_df['similarity'] = faiss_df['similarity'].astype(float)
    client: Optional[OpenAI] = None
    print(f'Prompt file: {paths.prompt_file}')
    print(f'Model: {MODEL} (temperature={TEMPERATURE})')
    for level in levels:
        level_df = faiss_df[faiss_df['level'] == level].copy()
        if level_df.empty:
            print(f'- {level}: no FAISS rows (skipping level)')
            continue
        level_df = level_df.sort_values(['query_user_id', 'rank'], kind='mergesort')
        query_ids = level_df['query_user_id'].unique().tolist()
        if limit is not None:
            query_ids = query_ids[:max(0, limit)]
        print(f'- {level}: {len(query_ids):,} queries to reason')
        for query_user_id in query_ids:
            key = (level, query_user_id)
            if key in done:
                continue
            true_candidate_id = query_user_id
            status = 'success'
            error = ''
            predicted_candidate_id = ''
            confidence = 0.0
            called_api = False
            try:
                top_df = level_df[level_df['query_user_id'] == query_user_id]
                cand_ids = top_df['candidate_user_id'].tolist()
                if not cand_ids:
                    raise ValueError('No candidates found for query in FAISS CSV.')
                query_path = paths.query_summaries_root / level / f'{query_user_id}_summary.json'
                if not query_path.exists():
                    raise FileNotFoundError(f'Query summary missing: {query_path}')
                query_summary = summary_obj_to_text(read_json(query_path))
                if not query_summary:
                    raise ValueError('Empty query summary text.')
                candidates: list[dict[str, object]] = []
                for (_, row) in top_df.iterrows():
                    cid = str(row['candidate_user_id'])
                    cpath = paths.candidate_summaries_dir / f'{cid}_summary.json'
                    if not cpath.exists():
                        raise FileNotFoundError(f'Candidate summary missing: {cpath}')
                    candidates.append({'candidate_user_id': cid, 'rank': int(row['rank']), 'score': float(row['similarity']), 'summary': summary_obj_to_text(read_json(cpath))})
                candidate_block = build_candidate_block(candidates)
                if client is None:
                    client = get_client(project_root)
                called_api = True
                out_obj = call_openai_reason(client=client, prompt_template=prompt_template, query_summary=query_summary, candidate_block=candidate_block)
                confidence = clamp01(out_obj.get('confidence', 0.0))
                (predicted_candidate_id, map_err) = resolve_predicted_candidate_id(out_obj, cand_ids)
                if map_err:
                    status = 'invalid_prediction'
                    error = map_err
            except Exception as e:
                status = 'error'
                error = str(e)
            is_correct = 1 if status == 'success' and predicted_candidate_id == true_candidate_id else 0
            append_row(paths.out_csv, {'level': level, 'query_user_id': query_user_id, 'predicted_candidate_id': predicted_candidate_id, 'true_candidate_id': true_candidate_id, 'confidence': float(confidence), 'is_correct': int(is_correct), 'status': status, 'error': error})
            done.add(key)
            if called_api:
                time.sleep(SLEEP_SECONDS)
    print(f'\nWrote predictions CSV: {paths.out_csv}')
if __name__ == '__main__':
    main()
