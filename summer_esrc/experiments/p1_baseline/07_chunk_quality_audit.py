from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
ROOT = Path(__file__).resolve().parents[2]
BSP = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from esrc.paths import summer_root
BUCKETS = [('1', lambda n: n == 1), ('2-3', lambda n: 2 <= n <= 3), ('4-6', lambda n: 4 <= n <= 6), ('7-10', lambda n: 7 <= n <= 10), ('11+', lambda n: n >= 11)]

def last_ok_jsonl(path: Path, id_keys: tuple[str, ...]=('user_id', 'query_user_id')) -> dict[str, dict]:
    by: dict[str, dict] = {}
    if not path.exists():
        return by
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get('status') != 'ok':
                continue
            uid = None
            for k in id_keys:
                if o.get(k):
                    uid = str(o[k])
                    break
            if uid:
                by[uid] = o
    return by

def load_gpt4o_summary(pool: str, uid: str) -> Optional[str]:
    if pool == 'hn':
        p = BSP / 'data' / 'extracted_summaries' / 'T8' / f'{uid}_summary.json'
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(obj, dict) and isinstance(obj.get('summary'), str):
            return obj['summary'].strip()
        return str(obj)
    p = BSP / 'data' / 'esrc' / 'pool_en' / 'summaries' / 'T8' / f'{uid}.txt'
    if p.exists():
        return p.read_text(encoding='utf-8').strip()
    return None

def trait_parts(summary: str) -> list[str]:
    parts = [p.strip() for p in summary.split(',')]
    return [p for p in parts if p]

def dup_rate(parts: list[str]) -> float:
    if not parts:
        return 0.0
    norm = [re.sub('\\s+', ' ', p.lower()) for p in parts]
    return 1.0 - len(set(norm)) / len(norm)

def looks_truncated(summary: str) -> bool:
    s = summary.rstrip()
    if not s:
        return False
    if s[-1] in '.!?;:"\')]':
        return False
    if s.endswith('-') or s.endswith('—'):
        return True
    if len(s) >= 1500 and ',' in s:
        last = s.split(',')[-1].strip()
        if last and last[-1].isalnum() and (len(last.split()) <= 2):
            return True
    if len(s) >= 3500 and s[-1].isalnum():
        return True
    return False

def bucket_label(n_chunks: int) -> str:
    for (label, fn) in BUCKETS:
        if fn(n_chunks):
            return label
    return 'other'

def mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=str(summer_root() / 'results' / 'runs' / 'p1_full_pool_overnight_20260805'))
    ap.add_argument('--out-dir', default=str(summer_root() / 'results' / 'p1_baseline'))
    ap.add_argument('--samples-per-bucket', type=int, default=2)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    tables = out_dir / 'tables'
    audit_dir = out_dir / 'chunk_audit'
    tables.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    bucket_rows: list[dict] = []
    sample_rows: list[dict] = []
    for pool in ('pool_en', 'hn'):
        meta = last_ok_jsonl(run_dir / pool / 'extract_meta.jsonl')
        reason = last_ok_jsonl(run_dir / pool / 'reason_predictions.jsonl', id_keys=('query_user_id', 'user_id'))
        extract_dir = run_dir / pool / 'extract'
        per_user: list[dict] = []
        for (uid, m) in meta.items():
            n_chunks = int(m.get('n_chunks') or (1 if not m.get('chunked') else 0) or 1)
            if not m.get('chunked'):
                n_chunks = 1
            local_path = extract_dir / f'{uid}.txt'
            local = local_path.read_text(encoding='utf-8').strip() if local_path.exists() else ''
            gpt = load_gpt4o_summary(pool, uid) or ''
            parts = trait_parts(local)
            gpt_parts = trait_parts(gpt) if gpt else []
            r = reason.get(uid, {})
            row = {'pool': pool, 'user_id': uid, 'n_chunks': n_chunks, 'bucket': bucket_label(n_chunks), 'n_words': m.get('n_words'), 'local_chars': len(local), 'gpt4o_chars': len(gpt) if gpt else None, 'char_ratio_local_over_gpt4o': len(local) / len(gpt) if gpt else None, 'local_traits': len(parts), 'gpt4o_traits': len(gpt_parts) if gpt_parts else None, 'dup_rate': round(dup_rate(parts), 4), 'truncated': looks_truncated(local), 'top1_correct': bool(r.get('correct')) if r else None, 'hit_at_15': bool(r.get('true_in_top15')) if r else None}
            per_user.append(row)
        user_csv = tables / f'chunk_audit_per_user_{pool}.csv'
        with user_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(per_user[0].keys()))
            w.writeheader()
            w.writerows(sorted(per_user, key=lambda r: (-r['n_chunks'], r['user_id'])))
        print(f'wrote {user_csv} n={len(per_user)}')
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for row in per_user:
            by_bucket[row['bucket']].append(row)
        for (label, _) in BUCKETS:
            rows = by_bucket.get(label, [])
            with_reason = [r for r in rows if r['top1_correct'] is not None]
            top1 = [r['top1_correct'] for r in with_reason]
            hit = [r['hit_at_15'] for r in with_reason]
            ratios = [r['char_ratio_local_over_gpt4o'] for r in rows if r['char_ratio_local_over_gpt4o'] is not None]
            trunc = [r['truncated'] for r in rows]
            dups = [r['dup_rate'] for r in rows]
            local_chars = [r['local_chars'] for r in rows]
            gpt_chars = [r['gpt4o_chars'] for r in rows if r['gpt4o_chars'] is not None]
            bucket_rows.append({'pool': pool, 'bucket': label, 'n_users': len(rows), 'n_with_reason': len(with_reason), 'top1_rate': round(sum(top1) / len(top1), 4) if top1 else None, 'top1_pct': round(100 * sum(top1) / len(top1), 1) if top1 else None, 'hit_at_15_rate': round(sum(hit) / len(hit), 4) if hit else None, 'hit_at_15_pct': round(100 * sum(hit) / len(hit), 1) if hit else None, 'mean_local_chars': round(mean(local_chars) or 0, 1), 'mean_gpt4o_chars': round(mean([float(x) for x in gpt_chars]) or 0, 1) if gpt_chars else None, 'mean_char_ratio': round(mean(ratios) or 0, 2) if ratios else None, 'pct_truncated': round(100 * sum((1 for t in trunc if t)) / len(trunc), 1) if trunc else None, 'mean_dup_rate': round(mean(dups) or 0, 4) if dups else None, 'mean_local_traits': round(mean([float(r['local_traits']) for r in rows]) or 0, 1)})
            ranked = sorted(rows, key=lambda r: (-r['n_chunks'], -r['local_chars']))
            for r in ranked[:args.samples_per_bucket]:
                uid = r['user_id']
                local = (extract_dir / f'{uid}.txt').read_text(encoding='utf-8').strip()
                gpt = load_gpt4o_summary(pool, uid) or ''
                sample_rows.append({**r, 'local_preview': local[:240], 'gpt4o_preview': gpt[:240]})
                if pool == 'hn' and label in ('7-10', '11+', '4-6'):
                    cmp = audit_dir / f'{uid}_COMPARE.md'
                    cmp.write_text(f"# {uid} ({pool}) n_chunks={r['n_chunks']} bucket={label}\n\n## gpt-4o archived ({len(gpt)} chars)\n\n{gpt}\n\n## local merge ({len(local)} chars, trunc={r['truncated']}, dup={r['dup_rate']}, traits≈{r['local_traits']})\n\n{local}\n", encoding='utf-8')
    out_buckets = tables / 'chunk_audit_by_bucket.csv'
    with out_buckets.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
        w.writeheader()
        w.writerows(bucket_rows)
    out_samples = tables / 'chunk_audit_samples.csv'
    with out_samples.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader()
        w.writerows(sample_rows)
    print('wrote', out_buckets)
    print('wrote', out_samples)
    print('\n=== Chunk → accuracy trend ===')
    for row in bucket_rows:
        print(f"{row['pool']:8} {row['bucket']:5} n={row['n_users']:3} top1={row['top1_pct']}% hit@15={row['hit_at_15_pct']}% char_ratio={row['mean_char_ratio']} trunc={row['pct_truncated']}%")
if __name__ == '__main__':
    main()
