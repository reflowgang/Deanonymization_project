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
PROMPT_REL_PATH = Path('prompts/summarization_lermen_g2.txt')
LEVELS_ALL: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
LOG_FIELDNAMES = ['T_level', 'user_id', 'input_path', 'output_path', 'status', 'error', 'input_words', 'output_words', 'model', 'prompt_file']

@dataclass(frozen=True)
class Paths:
    truncated_root: Path
    summaries_root: Path
    extract_log_csv: Path
    prompt_file: Path

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def parse_args(argv: Optional[list[str]]=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Extract summaries from POOL-EN truncated query profiles using a prompt file.')
    p.add_argument('--levels', nargs='+', default=LEVELS_ALL, help='Truncation levels to process, e.g. --levels T1 T8 (default: all).')
    p.add_argument('--limit', type=int, default=None, help='Process only the first N files per level (for testing).')
    return p.parse_args(argv)

def normalize_levels(levels: Iterable[str]) -> list[str]:
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

def read_profile_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')

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

def log_row_base(level: str, user_id: str, in_path: Path, out_path: Path, prompt_file: str) -> dict[str, object]:
    return {'T_level': level, 'user_id': user_id, 'input_path': str(in_path), 'output_path': str(out_path), 'model': MODEL, 'prompt_file': prompt_file}

def main() -> None:
    args = parse_args()
    levels = normalize_levels(args.levels)
    limit = args.limit
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(truncated_root=project_root / 'data/esrc/pool_en/truncated_queries', summaries_root=project_root / 'data/esrc/pool_en/summaries', extract_log_csv=project_root / 'results/tables/pool_en_extract_log.csv', prompt_file=project_root / PROMPT_REL_PATH)
    prompt_file_log = str(PROMPT_REL_PATH).replace('\\', '/')
    for lvl in LEVELS_ALL:
        _ensure_dir(paths.summaries_root / lvl)
    prompt_template = load_prompt_template(paths.prompt_file)
    print(f'Truncated inputs: {paths.truncated_root}')
    print(f'Summaries output: {paths.summaries_root}')
    print(f'Prompt file:      {paths.prompt_file}')
    print(f'Model: {MODEL} (temperature={TEMPERATURE})')
    client = get_client(project_root)
    for level in levels:
        in_dir = paths.truncated_root / level
        out_dir = paths.summaries_root / level
        if not in_dir.exists():
            print(f'- {level}: input directory missing, skipping: {in_dir}')
            continue
        files = sorted(in_dir.glob('*.txt'))
        if limit is not None:
            files = files[:max(0, limit)]
        print(f'- {level}: {len(files):,} input files')
        for in_path in try_tqdm(files, total=len(files), desc=level):
            user_id = in_path.stem
            out_path = out_dir / f'{user_id}.txt'
            base = log_row_base(level, user_id, in_path, out_path, prompt_file_log)
            if should_skip_existing_output(out_path):
                profile_text = read_profile_text(in_path)
                summary_text = out_path.read_text(encoding='utf-8')
                append_log_row(paths.extract_log_csv, {**base, 'status': 'skipped_existing', 'error': '', 'input_words': int(count_words(profile_text)), 'output_words': int(count_words(summary_text))})
                continue
            try:
                profile_text = read_profile_text(in_path)
                input_words = count_words(profile_text)
                if input_words == 0:
                    append_log_row(paths.extract_log_csv, {**base, 'status': 'error', 'error': 'empty_input_profile', 'input_words': 0, 'output_words': 0})
                    continue
                summary_text = call_openai_summary(client, prompt_template, profile_text)
                output_words = count_words(summary_text)
                _ensure_parent_dir(out_path)
                out_path.write_text(summary_text + ('\n' if summary_text else ''), encoding='utf-8')
                append_log_row(paths.extract_log_csv, {**base, 'status': 'ok', 'error': '', 'input_words': int(input_words), 'output_words': int(output_words)})
                time.sleep(SLEEP_SECONDS)
            except Exception as e:
                append_log_row(paths.extract_log_csv, {**base, 'status': 'error', 'error': str(e), 'input_words': int(count_words(read_profile_text(in_path))) if in_path.exists() else 0, 'output_words': 0})
                time.sleep(SLEEP_SECONDS)
    print(f'\nWrote/updated log: {paths.extract_log_csv}')
    print('Done.')
if __name__ == '__main__':
    main()
