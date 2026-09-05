from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from openai import OpenAI
EXTRACT_MODEL = 'qwen3.5-4b'
CONTEXT_LIMIT = 16384
WORD_MIN = 10000
WORD_MAX = 15000
BUCKET_CENTERS = [10000, 11000, 12000, 13000, 14000, 15000]
PER_BUCKET = 2
N_TOTAL = 10
LEVELS = {'T5', 'T6', 'T7'}

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f'Prompt file not found: {path}')
    return path.read_text(encoding='utf-8')

def read_profiles(stats_csv: Path, levels: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with stats_csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('T_level') in levels:
                rows.append(row)
    return rows

def pick_spread_buckets(rows: list[dict[str, str]], *, word_min: int, word_max: int, bucket_centers: list[int], per_bucket: int, n_total: int) -> list[dict[str, object]]:
    candidates = [r for r in rows if word_min <= int(r['n_words']) <= word_max]
    if not candidates:
        raise RuntimeError(f'No profiles in word range [{word_min}, {word_max}]')
    picked: list[dict[str, object]] = []
    picked_ids: set[tuple[str, str]] = set()
    for center in bucket_centers:
        if len(picked) >= n_total:
            break
        pool = [r for r in candidates if (r['T_level'], r['user_id']) not in picked_ids]
        pool.sort(key=lambda r: abs(int(r['n_words']) - center))
        for r in pool[:per_bucket]:
            if len(picked) >= n_total:
                break
            picked.append({**r, 'bucket_target_words': center})
            picked_ids.add((r['T_level'], r['user_id']))
    if len(picked) < n_total:
        remaining = [r for r in candidates if (r['T_level'], r['user_id']) not in picked_ids]
        remaining.sort(key=lambda r: int(r['n_words']))
        for r in remaining:
            if len(picked) >= n_total:
                break
            picked.append({**r, 'bucket_target_words': ''})
            picked_ids.add((r['T_level'], r['user_id']))
    picked.sort(key=lambda r: int(r['n_words']))
    return picked[:n_total]

def build_user_content(prompt_template: str, profile_text: str) -> str:
    if '{profile_text}' in prompt_template:
        return prompt_template.replace('{profile_text}', profile_text)
    return prompt_template.rstrip() + '\n\nCOMMENTS:\n' + profile_text

def is_context_overflow(msg: str) -> bool:
    msg_l = msg.lower()
    return 'maximum context length' in msg_l or 'prompt contains at least' in msg_l or 'your prompt contains' in msg_l or ('context length' in msg_l and 'tokens' in msg_l)

def measure_prompt_tokens(client: OpenAI, user_content: str) -> tuple[str, int | None, int | None, str]:
    try:
        resp = client.chat.completions.create(model=EXTRACT_MODEL, temperature=0.0, max_tokens=1, messages=[{'role': 'user', 'content': user_content}])
        usage = getattr(resp, 'usage', None)
        prompt_tokens = int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else None
        total_tokens = int(usage.total_tokens) if usage and usage.total_tokens is not None else None
        return ('ok', prompt_tokens, total_tokens, '')
    except Exception as e:
        msg = str(e)
        if is_context_overflow(msg):
            return ('exceeds_limit', None, None, '')
        return ('error', None, None, msg[:300])

def main() -> int:
    root = project_root()
    stats_csv = root / 'results/tables/pool_en_truncation_stats.csv'
    prompt_path = root / 'prompts/summarization_lermen_g2.txt'
    truncated_root = root / 'data/esrc/pool_en/truncated_queries'
    out_csv = root / 'summer_esrc/results/token_counts_bracket_10k_15k.csv'
    out_meta = root / 'summer_esrc/results/token_counts_bracket_10k_15k_meta.json'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get('VLLM_BASE_URL')
    api_key = os.environ.get('VLLM_API_KEY')
    if not base_url:
        raise RuntimeError('Missing VLLM_BASE_URL')
    if not api_key:
        raise RuntimeError('Missing VLLM_API_KEY')
    prompt_template = load_prompt_template(prompt_path)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    template_content = build_user_content(prompt_template, '')
    (tpl_status, tpl_tokens, tpl_total, tpl_err) = measure_prompt_tokens(client, template_content)
    print('Template-only diagnostic (empty profile_text):')
    print(f'  status={tpl_status} prompt_tokens={tpl_tokens} total_tokens={tpl_total}' + (f' error={tpl_err!r}' if tpl_err else ''))
    print()
    candidates = read_profiles(stats_csv, LEVELS)
    selected = pick_spread_buckets(candidates, word_min=WORD_MIN, word_max=WORD_MAX, bucket_centers=BUCKET_CENTERS, per_bucket=PER_BUCKET, n_total=N_TOTAL)
    print(f'Bracket sample: {N_TOTAL} profiles, words in [{WORD_MIN:,}, {WORD_MAX:,}], levels {sorted(LEVELS)}, model={EXTRACT_MODEL}, limit={CONTEXT_LIMIT}')
    print('-' * 100)
    rows_out: list[dict[str, object]] = []
    for (i, row) in enumerate(selected, start=1):
        lvl = str(row['T_level'])
        uid = str(row['user_id'])
        n_words = int(row['n_words'])
        bucket_target = row.get('bucket_target_words', '')
        in_path = truncated_root / lvl / f'{uid}.txt'
        if not in_path.exists():
            raise FileNotFoundError(f'Missing truncated input file: {in_path}')
        profile_text = in_path.read_text(encoding='utf-8')
        user_content = build_user_content(prompt_template, profile_text)
        (status, prompt_tokens, total_tokens, err_str) = measure_prompt_tokens(client, user_content)
        body_tokens_est = None
        tokens_per_word = None
        if status == 'ok' and prompt_tokens is not None and (tpl_tokens is not None):
            body_tokens_est = prompt_tokens - tpl_tokens
            if n_words > 0:
                tokens_per_word = round(body_tokens_est / n_words, 4)
        rows_out.append({'rank': i, 'T_level': lvl, 'user_id': uid, 'n_words': n_words, 'bucket_target_words': bucket_target, 'status': status, 'prompt_tokens': prompt_tokens, 'template_overhead_tokens': tpl_tokens, 'body_tokens_est': body_tokens_est, 'tokens_per_word_est': tokens_per_word, 'total_tokens': total_tokens, 'error': err_str})
        print(f'{i:>2}. {lvl} {uid} words={n_words:,} bucket~{bucket_target} status={status} prompt_tokens={prompt_tokens} body_est={body_tokens_est} tok/word={tokens_per_word}')
    fieldnames = ['rank', 'T_level', 'user_id', 'n_words', 'bucket_target_words', 'status', 'prompt_tokens', 'template_overhead_tokens', 'body_tokens_est', 'tokens_per_word_est', 'total_tokens', 'error']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    ok_rows = [r for r in rows_out if r['status'] == 'ok' and r['tokens_per_word_est']]
    tpw_vals = [float(r['tokens_per_word_est']) for r in ok_rows]
    meta = {'extract_model': EXTRACT_MODEL, 'context_limit_tokens': CONTEXT_LIMIT, 'word_range': [WORD_MIN, WORD_MAX], 'levels': sorted(LEVELS), 'bucket_centers': BUCKET_CENTERS, 'template_only': {'status': tpl_status, 'prompt_tokens': tpl_tokens, 'total_tokens': tpl_total, 'error': tpl_err}, 'n_profiles': len(rows_out), 'n_ok': sum((1 for r in rows_out if r['status'] == 'ok')), 'n_exceeds_limit': sum((1 for r in rows_out if r['status'] == 'exceeds_limit')), 'tokens_per_word_est': {'mean': round(sum(tpw_vals) / len(tpw_vals), 4) if tpw_vals else None, 'min': min(tpw_vals) if tpw_vals else None, 'max': max(tpw_vals) if tpw_vals else None}}
    out_meta.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print('-' * 100)
    print(f'Wrote: {out_csv}')
    print(f'Wrote: {out_meta}')
    if tpw_vals:
        print(f"tokens_per_word_est (ok only): mean={meta['tokens_per_word_est']['mean']} min={meta['tokens_per_word_est']['min']} max={meta['tokens_per_word_est']['max']}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
