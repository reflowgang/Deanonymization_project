from __future__ import annotations
import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from dotenv import load_dotenv
from openai import OpenAI
MODEL = 'gpt-4o-mini'
TEMPERATURE = 0
SLEEP_SECONDS = 0.2
MAX_INPUT_WORDS = 25000
PROMPT_REL_PATH = Path('prompts/summarization_lermen_g2.txt')
LOG_FIELDNAMES = ['user_id', 'input_path', 'output_path', 'status', 'error', 'original_input_words', 'sent_input_words', 'was_truncated', 'input_words', 'output_words', 'model', 'prompt_file']

@dataclass(frozen=True)
class Paths:
    candidate_profiles_dir: Path
    candidate_summaries_dir: Path
    extract_log_csv: Path
    prompt_file: Path

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_args(argv: Optional[list[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Extract summaries from POOL-EN candidate profiles using a prompt file.')
    p.add_argument('--limit', type=int, default=None, help='Process only the first N candidate files (for testing).')
    return p.parse_args(argv)

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

def build_user_prompt(template: str, profile_text: str) -> str:
    if '{profile_text}' in template:
        return template.replace('{profile_text}', profile_text)
    return template.rstrip() + '\n\nCOMMENTS:\n' + profile_text

def try_tqdm(iterable: Iterable[Path], total: int, desc: str) -> Iterable[Path]:
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable

def count_words(text: str) -> int:
    return len(text.split())

def truncate_to_first_n_words(text: str, max_words: int) -> tuple[str, int, bool]:
    words = text.split()
    original_words = len(words)
    if original_words <= max_words:
        return (text, original_words, False)
    truncated = ' '.join(words[:max_words])
    return (truncated, original_words, True)

def should_skip_existing_output(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        return out_path.stat().st_size > 0
    except OSError:
        return False

def call_openai_summary(client: OpenAI, prompt_template: str, profile_text: str) -> str:
    user_content = build_user_prompt(prompt_template, profile_text)
    resp = client.chat.completions.create(model=MODEL, temperature=TEMPERATURE, messages=[{'role': 'user', 'content': user_content}])
    content = resp.choices[0].message.content or ''
    return content.strip()

def append_log_row(log_csv: Path, row: dict[str, object]) -> None:
    _ensure_parent_dir(log_csv)
    file_exists = log_csv.exists()
    with log_csv.open('a', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def log_row_base(user_id: str, in_path: Path, out_path: Path, prompt_file: str, original_input_words: int, sent_input_words: int, was_truncated: bool) -> dict[str, object]:
    return {'user_id': user_id, 'input_path': str(in_path), 'output_path': str(out_path), 'original_input_words': int(original_input_words), 'sent_input_words': int(sent_input_words), 'was_truncated': int(was_truncated), 'input_words': int(original_input_words), 'model': MODEL, 'prompt_file': prompt_file}

def main() -> None:
    args = parse_args()
    limit = args.limit
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(candidate_profiles_dir=project_root / 'data/esrc/pool_en/candidate_profiles', candidate_summaries_dir=project_root / 'data/esrc/pool_en/candidate_summaries', extract_log_csv=project_root / 'results/tables/pool_en_candidate_extract_log.csv', prompt_file=project_root / PROMPT_REL_PATH)
    prompt_file_log = str(PROMPT_REL_PATH).replace('\\', '/')
    _ensure_dir(paths.candidate_summaries_dir)
    _ensure_parent_dir(paths.extract_log_csv)
    prompt_template = load_prompt_template(paths.prompt_file)
    print(f'Candidate inputs:  {paths.candidate_profiles_dir}')
    print(f'Summaries output:  {paths.candidate_summaries_dir}')
    print(f'Prompt file:       {paths.prompt_file}')
    print(f'Model: {MODEL} (temperature={TEMPERATURE})')
    print(f'Max input words (sent to API): {MAX_INPUT_WORDS:,}')
    if not paths.candidate_profiles_dir.exists():
        raise FileNotFoundError(f'Candidate profiles directory not found: {paths.candidate_profiles_dir}')
    files = sorted(paths.candidate_profiles_dir.glob('*.txt'))
    if limit is not None:
        files = files[:max(0, limit)]
    client = get_client(project_root)
    for in_path in try_tqdm(files, total=len(files), desc='candidate'):
        user_id = in_path.stem
        out_path = paths.candidate_summaries_dir / f'{user_id}.txt'
        profile_text = in_path.read_text(encoding='utf-8') if in_path.exists() else ''
        (sent_text, original_input_words, was_truncated) = truncate_to_first_n_words(profile_text, MAX_INPUT_WORDS)
        sent_input_words = count_words(sent_text)
        base = log_row_base(user_id, in_path, out_path, prompt_file_log, original_input_words, sent_input_words, was_truncated)
        if should_skip_existing_output(out_path):
            summary_text = out_path.read_text(encoding='utf-8')
            append_log_row(paths.extract_log_csv, {**base, 'status': 'skipped_existing', 'error': '', 'output_words': int(count_words(summary_text))})
            continue
        try:
            if original_input_words == 0:
                append_log_row(paths.extract_log_csv, {**base, 'status': 'error', 'error': 'empty_input_profile', 'output_words': 0})
                continue
            summary_text = call_openai_summary(client, prompt_template, sent_text)
            output_words = count_words(summary_text)
            _ensure_parent_dir(out_path)
            out_path.write_text(summary_text + ('\n' if summary_text else ''), encoding='utf-8')
            append_log_row(paths.extract_log_csv, {**base, 'status': 'ok', 'error': '', 'output_words': int(output_words)})
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            append_log_row(paths.extract_log_csv, {**base, 'status': 'error', 'error': str(e), 'output_words': 0})
            time.sleep(SLEEP_SECONDS)
    print(f'\nWrote/updated log: {paths.extract_log_csv}')
    print('Done.')
if __name__ == '__main__':
    main()
