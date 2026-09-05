from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.config import load_dotenv_files
from esrc.extract import CHUNK_GATE_TOKENS, extract_profile, load_prompt_template
from esrc.generate import get_client
from esrc.manifests import git_commit, write_manifest
from esrc.paths import bsp_prompt_summarization, bsp_truncated_query, summer_prompt_extract_merge, summer_root
DEFAULT_EXTRACT_MODEL = 'qwen3.5-4b'
DEFAULT_CASES: list[tuple[str, str, str]] = [('T7', 'user_0cc626ee', 'under_gate_single_chunk'), ('T6', 'user_378003e3', 'borderline_over_gate'), ('T7', 'user_c709e252', 'last_fit_before_overflow'), ('T7', 'user_e9d1886c', 'first_overflow'), ('T7', 'user_daf37424', 'top10_longest'), ('T8', 'user_daf37424', 'top10_longest_t8')]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Test Extract chunking on long profiles')
    p.add_argument('--model', default=os.getenv('VLLM_EXTRACT_MODEL', DEFAULT_EXTRACT_MODEL))
    p.add_argument('--out-dir', default=None, help='Default: results/runs/p1_chunking_test_<timestamp>/')
    p.add_argument('--max-tokens', type=int, default=1024)
    return p.parse_args()

def main() -> int:
    load_dotenv_files(summer_root())
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else summer_root() / 'results' / 'runs' / f"p1_chunking_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_tpl = load_prompt_template(bsp_prompt_summarization())
    merge_tpl = load_prompt_template(summer_prompt_extract_merge())
    client = get_client()
    rows: list[dict] = []
    print(f'model={args.model} gate={CHUNK_GATE_TOKENS} cases={len(DEFAULT_CASES)}')
    for (level, uid, label) in DEFAULT_CASES:
        path = bsp_truncated_query(level, uid)
        if not path.exists():
            print(f'SKIP missing {path}')
            continue
        text = path.read_text(encoding='utf-8')
        n_words = len(text.split())
        print(f'\n--- {label} {level} {uid} ({n_words} words) ---')
        try:
            result = extract_profile(text, client=client, model=args.model, extract_template=extract_tpl, merge_template=merge_tpl, max_tokens=args.max_tokens, enable_thinking=False)
            row = {'label': label, 'T_level': level, 'user_id': uid, 'n_words': n_words, 'status': 'ok', 'chunked': result.chunked, 'n_chunks': result.n_chunks, 'method': result.method, 'chunk_sizes': list(result.chunk_sizes), 'probe_tokens_full': result.probe_tokens_full, 'probe_tokens_per_chunk': list(result.probe_tokens_per_chunk), 'summary_chars': len(result.summary), 'summary_preview': result.summary[:240]}
            (out_dir / f'{uid}_{level}_summary.txt').write_text(result.summary + '\n', encoding='utf-8')
            print(f'ok chunked={result.chunked} n_chunks={result.n_chunks} method={result.method} probe_full={result.probe_tokens_full} sizes={result.chunk_sizes}')
        except Exception as exc:
            row = {'label': label, 'T_level': level, 'user_id': uid, 'n_words': n_words, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            print(f"ERR {uid}: {row['error']}")
        rows.append(row)
    out_csv = out_dir / 'chunking_test_results.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
    write_manifest(out_dir / 'manifest.json', {'task': 'p1_chunking_test', 'model': args.model, 'chunk_gate_tokens': CHUNK_GATE_TOKENS, 'n_cases': len(rows), 'n_ok': sum((1 for r in rows if r.get('status') == 'ok')), 'git_commit': git_commit(summer_root()), 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    print(f'\nWrote {out_csv}')
    return 0 if all((r.get('status') == 'ok' for r in rows)) else 1
if __name__ == '__main__':
    raise SystemExit(main())
