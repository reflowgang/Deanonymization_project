from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSIFY_MODEL, CONTENT_TYPES, RANDOM_SEED, paths
from io_utils import iter_sorted_comments, load_query_user_ids, resolve_run_paths, select_pilot_users
TEMPERATURE = 0
SLEEP_SECONDS = 0.2
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES_SINGLE = 3
FORCED_TOPICAL_STATUS = 'forced_topical_after_failure'
SINGLE_PROMPT_FILE_LOG = 'inline_single_comment_prompt'
SINGLE_COMMENT_PROMPT = 'Classify this Reddit comment as exactly one letter.\n\nP = personal disclosure about the author\'s life\nO = opinion or value statement\nT = topical discussion without personal disclosure\n\nComment:\n{body}\n\nRespond with ONLY valid JSON: {{"label": "P"}} or {{"label": "O"}} or {{"label": "T"}}'
CLASSIFICATION_FIELDS = ['user_id', 'comment_index', 'timestamp', 'source_line', 'label', 'body']
LOG_FIELDS = ['user_id', 'batch_start_index', 'batch_end_index', 'batch_size', 'attempt', 'mode', 'status', 'error', 'model', 'prompt_file', 'n_labels_returned']

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Classify query comments as P/O/T.')
    p.add_argument('--pilot', action='store_true', help='Pilot run with separate output paths.')
    p.add_argument('--pilot-users', type=int, default=30, help='Users in pilot mode.')
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--limit-users', type=int, default=None, help='Cap users (testing).')
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--max-retries-single', type=int, default=DEFAULT_MAX_RETRIES_SINGLE, help='Max API attempts per comment in simple single-comment recovery (default: 3).')
    return p.parse_args()

def load_api_key(project_root: Path) -> None:
    env_path = project_root / 'api_key.env'
    if not env_path.exists():
        env_path = project_root / '.env'
    load_dotenv(env_path, override=False)
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError(f'OPENAI_API_KEY not found. Expected it in {env_path}.')

def get_client(project_root: Path) -> OpenAI:
    load_api_key(project_root)
    return OpenAI()

def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding='utf-8')

def build_comment_block(batch: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in batch:
        parts.append(f"[{item['comment_index']}] {item['body']}")
    return '\n\n'.join(parts)

def build_user_prompt(template: str, batch: list[dict[str, Any]]) -> str:
    if '{comment_block}' not in template:
        raise ValueError('Prompt template must contain {comment_block}')
    return template.replace('{comment_block}', build_comment_block(batch))

def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip().upper()
    if label in CONTENT_TYPES:
        return label
    if label.startswith('P'):
        return 'P'
    if label.startswith('O'):
        return 'O'
    if label.startswith('T'):
        return 'T'
    return None

def parse_classifications(content: str, batch: list[dict[str, Any]]) -> dict[int, str]:
    expected_indices = [int(item['comment_index']) for item in batch]
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError('Model returned non-object JSON.')
    raw_items = obj.get('classifications')
    if not isinstance(raw_items, list):
        raise ValueError('Missing classifications array in JSON.')
    parsed: dict[int, str] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        idx = item.get('comment_index')
        if idx is None:
            continue
        idx_int = int(idx)
        label = normalize_label(item.get('label'))
        if label is None:
            raise ValueError(f"Invalid label for comment_index={idx_int}: {item.get('label')!r}")
        parsed[idx_int] = label
    relative_keys = set(range(1, len(batch) + 1))
    if parsed and set(parsed.keys()) == relative_keys:
        parsed = {expected_indices[i - 1]: parsed[i] for i in range(1, len(batch) + 1)}
    missing = [idx for idx in expected_indices if idx not in parsed]
    if missing:
        raise ValueError(f"Missing labels for comment indices: {missing[:5]}{('...' if len(missing) > 5 else '')}")
    return parsed

def call_classifier(client: OpenAI, model: str, prompt_template: str, batch: list[dict[str, Any]]) -> dict[int, str]:
    user_prompt = build_user_prompt(prompt_template, batch)
    resp = client.chat.completions.create(model=model, temperature=TEMPERATURE, messages=[{'role': 'user', 'content': user_prompt}], response_format={'type': 'json_object'})
    content = resp.choices[0].message.content or ''
    return parse_classifications(content, batch)

def parse_single_label(content: str) -> str:
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError('Model returned non-object JSON.')
    label = normalize_label(obj.get('label'))
    if label is None:
        raise ValueError(f"Invalid single-comment label: {obj.get('label')!r}")
    return label

def call_single_comment_simple(client: OpenAI, model: str, comment: dict[str, Any]) -> str:
    body = str(comment['body'])
    user_prompt = SINGLE_COMMENT_PROMPT.format(body=body)
    resp = client.chat.completions.create(model=model, temperature=TEMPERATURE, messages=[{'role': 'user', 'content': user_prompt}], response_format={'type': 'json_object'})
    content = resp.choices[0].message.content or ''
    return parse_single_label(content)

def classify_missing_comments(client: OpenAI, model: str, user_id: str, missing_comments: list[dict[str, Any]], max_retries_single: int) -> tuple[dict[int, str], list[dict[str, object]]]:
    labels: dict[int, str] = {}
    log_rows: list[dict[str, object]] = []
    for comment in missing_comments:
        idx = int(comment['comment_index'])
        label: Optional[str] = None
        last_error = ''
        for attempt in range(1, max_retries_single + 1):
            try:
                label = call_single_comment_simple(client, model, comment)
                log_rows.append({'user_id': user_id, 'batch_start_index': idx, 'batch_end_index': idx, 'batch_size': 1, 'attempt': attempt, 'mode': 'single_simple', 'status': 'ok', 'error': '', 'model': model, 'prompt_file': SINGLE_PROMPT_FILE_LOG, 'n_labels_returned': 1})
                break
            except Exception as e:
                last_error = str(e)
                log_rows.append({'user_id': user_id, 'batch_start_index': idx, 'batch_end_index': idx, 'batch_size': 1, 'attempt': attempt, 'mode': 'single_simple', 'status': 'error', 'error': last_error, 'model': model, 'prompt_file': SINGLE_PROMPT_FILE_LOG, 'n_labels_returned': 0})
                time.sleep(SLEEP_SECONDS)
        if label is None:
            label = 'T'
            log_rows.append({'user_id': user_id, 'batch_start_index': idx, 'batch_end_index': idx, 'batch_size': 1, 'attempt': max_retries_single, 'mode': 'single_simple', 'status': FORCED_TOPICAL_STATUS, 'error': last_error, 'model': model, 'prompt_file': SINGLE_PROMPT_FILE_LOG, 'n_labels_returned': 1})
        labels[idx] = label
        time.sleep(SLEEP_SECONDS)
    return (labels, log_rows)

def ensure_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def append_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    with path.open('a', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

def append_classification_row(classifications_csv: Path, user_id: str, comment: dict[str, Any], label: str) -> None:
    append_row(classifications_csv, CLASSIFICATION_FIELDS, {'user_id': user_id, 'comment_index': int(comment['comment_index']), 'timestamp': int(comment['timestamp']), 'source_line': int(comment['source_line']), 'label': label, 'body': str(comment['body'])})

def persist_batch_results(classifications_csv: Path, log_csv: Path, user_id: str, batch: list[dict[str, Any]], labels: dict[int, str], log_rows: list[dict[str, object]], already: set[int]) -> int:
    api_calls = 0
    for log_row in log_rows:
        append_row(log_csv, LOG_FIELDS, log_row)
        if log_row['status'] == 'ok':
            api_calls += 1
    for comment in batch:
        idx = int(comment['comment_index'])
        if idx not in labels or idx in already:
            continue
        append_classification_row(classifications_csv, user_id, comment, labels[idx])
        already.add(idx)
    return api_calls

def load_existing_indices(classifications_csv: Path) -> dict[str, set[int]]:
    if not classifications_csv.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(classifications_csv)
    if df.empty or 'user_id' not in df.columns:
        return {}
    existing: dict[str, set[int]] = {}
    for (user_id, sub) in df.groupby('user_id'):
        existing[str(user_id)] = set(sub['comment_index'].astype(int).tolist())
    return existing

def classify_batch_with_retry(client: OpenAI, model: str, prompt_template: str, user_id: str, batch: list[dict[str, Any]], prompt_file_log: str) -> tuple[dict[int, str], list[dict[str, object]]]:
    start_idx = int(batch[0]['comment_index'])
    end_idx = int(batch[-1]['comment_index'])
    log_rows: list[dict[str, object]] = []
    for attempt in (1, 2):
        try:
            labels = call_classifier(client, model, prompt_template, batch)
            log_rows.append({'user_id': user_id, 'batch_start_index': start_idx, 'batch_end_index': end_idx, 'batch_size': len(batch), 'attempt': attempt, 'mode': 'batch', 'status': 'ok', 'error': '', 'model': model, 'prompt_file': prompt_file_log, 'n_labels_returned': len(labels)})
            return (labels, log_rows)
        except Exception as e:
            log_rows.append({'user_id': user_id, 'batch_start_index': start_idx, 'batch_end_index': end_idx, 'batch_size': len(batch), 'attempt': attempt, 'mode': 'batch', 'status': 'error', 'error': str(e), 'model': model, 'prompt_file': prompt_file_log, 'n_labels_returned': 0})
            time.sleep(SLEEP_SECONDS)
    labels: dict[int, str] = {}
    for comment in batch:
        idx = int(comment['comment_index'])
        try:
            batch_labels = call_classifier(client, model, prompt_template, [comment])
            labels[idx] = batch_labels[idx]
            log_rows.append({'user_id': user_id, 'batch_start_index': idx, 'batch_end_index': idx, 'batch_size': 1, 'attempt': 1, 'mode': 'single_fallback', 'status': 'ok', 'error': '', 'model': model, 'prompt_file': prompt_file_log, 'n_labels_returned': 1})
        except Exception as e:
            log_rows.append({'user_id': user_id, 'batch_start_index': idx, 'batch_end_index': idx, 'batch_size': 1, 'attempt': 1, 'mode': 'single_fallback', 'status': 'error', 'error': str(e), 'model': model, 'prompt_file': prompt_file_log, 'n_labels_returned': 0})
        time.sleep(SLEEP_SECONDS)
    return (labels, log_rows)

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    run_paths = resolve_run_paths(pilot=args.pilot)
    p = paths()
    user_ids = load_query_user_ids()
    if args.pilot:
        user_ids = select_pilot_users(user_ids, n_users=args.pilot_users, seed=args.seed)
    if args.limit_users is not None:
        user_ids = user_ids[:args.limit_users]
    batch_size = max(1, int(args.batch_size))
    max_retries_single = max(1, int(args.max_retries_single))
    prompt_template = load_prompt_template(p['classify_prompt'])
    prompt_file_log = 'prompts/content_type_classification.txt'
    classifications_csv = run_paths['classifications_csv']
    log_csv = run_paths['classify_log_csv']
    ensure_csv(classifications_csv, CLASSIFICATION_FIELDS)
    ensure_csv(log_csv, LOG_FIELDS)
    existing_indices = load_existing_indices(classifications_csv)
    print(f"Mode: {('pilot' if args.pilot else 'production')}")
    print(f'Users selected: {len(user_ids):,}')
    print(f'Users with partial progress: {sum((1 for uid in user_ids if existing_indices.get(uid))):,}')
    print(f'Batch size: {batch_size}')
    print(f'Max single-comment retries: {max_retries_single}')
    print(f'Model: {CLASSIFY_MODEL}')
    print(f'Output classifications: {classifications_csv}')
    print(f'Output log: {log_csv}')
    client = get_client(project_root)
    total_batches = 0
    total_api_calls = 0
    users_completed = 0
    for user_id in user_ids:
        jsonl_path = p['raw_query_jsonl'] / f'{user_id}_query.jsonl'
        if not jsonl_path.exists():
            raise FileNotFoundError(f'Missing query JSONL: {jsonl_path}')
        comments = iter_sorted_comments(jsonl_path)
        if len(comments) != 500:
            print(f'Warning: {user_id} has {len(comments)} comments (expected 500).')
        already = set(existing_indices.get(user_id, set()))
        for start in range(0, len(comments), batch_size):
            batch = comments[start:start + batch_size]
            if all((int(c['comment_index']) in already for c in batch)):
                continue
            batch = [c for c in batch if int(c['comment_index']) not in already]
            if not batch:
                continue
            total_batches += 1
            (labels, log_rows) = classify_batch_with_retry(client=client, model=CLASSIFY_MODEL, prompt_template=prompt_template, user_id=user_id, batch=batch, prompt_file_log=prompt_file_log)
            total_api_calls += persist_batch_results(classifications_csv, log_csv, user_id, batch, labels, log_rows, already)
            missing_comments = [c for c in batch if int(c['comment_index']) not in already]
            if missing_comments:
                (recovery_labels, recovery_logs) = classify_missing_comments(client=client, model=CLASSIFY_MODEL, user_id=user_id, missing_comments=missing_comments, max_retries_single=max_retries_single)
                total_api_calls += persist_batch_results(classifications_csv, log_csv, user_id, missing_comments, recovery_labels, recovery_logs, already)
            time.sleep(SLEEP_SECONDS)
        users_completed += 1
        print(f'- classified {user_id}: {len(comments):,} comments')
    print(f'\nCompleted classification pass for {users_completed:,} users.')
    print(f'Batch groups processed: {total_batches:,}')
    print(f'Successful API calls logged: {total_api_calls:,}')
if __name__ == '__main__':
    main()
