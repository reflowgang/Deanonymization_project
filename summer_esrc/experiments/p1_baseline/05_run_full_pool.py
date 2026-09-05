from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.config import load_dotenv_files
from esrc.embed import embed_texts
from esrc.extract import extract_profile, load_prompt_template
from esrc.generate import ContextLengthExceededError, PermanentRequestError, generate, get_client
from esrc.manifests import git_commit, write_manifest
from esrc.paths import bsp_candidate_embeddings_index_csv, bsp_candidate_embeddings_npy, bsp_hn_candidate_summaries, bsp_hn_pool_manifest, bsp_hn_truncated_dir, bsp_hn_user_mapping, bsp_pool_en, bsp_pool_en_candidate_summaries, bsp_prompt_record_selection, bsp_prompt_summarization, bsp_truncated_query, list_hn_query_user_ids, list_pool_en_query_user_ids, load_hn_candidate_summary_text, load_hn_query_profile_text, summer_prompt_extract_merge, summer_root
from esrc.reason_prompt import build_candidate_block, build_user_prompt, extract_json_object_for_reason, JsonParseError, load_prompt_template as load_reason_template, resolve_predicted_candidate_id
from esrc.resume import load_last_ok_by_user, load_ok_user_ids, load_resume_skip_user_ids
from esrc.search import search_top_k
DEFAULT_EXTRACT_MODEL = 'qwen3.5-4b'
DEFAULT_REASON_MODEL = 'qwen3.6-35b-a3b-nvfp4'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Full-pool Extract → Search → Reason')
    p.add_argument('--pool', choices=('pool_en', 'hn', 'both'), default='both', help='Which pool(s) to run (default: both)')
    p.add_argument('--level', default='T8')
    p.add_argument('--extract-model', default=os.getenv('VLLM_EXTRACT_MODEL', DEFAULT_EXTRACT_MODEL))
    p.add_argument('--reason-model', default=os.getenv('VLLM_REASON_MODEL', DEFAULT_REASON_MODEL))
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--limit', type=int, default=None, help='Cap query users per pool')
    p.add_argument('--user-ids', type=str, default=None, help='Comma-separated query_user_ids (smoke / subset), applied per pool')
    p.add_argument('--phase', choices=('all', 'extract', 'search', 'reason'), default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--out-dir', default=None, help='Parent run dir; creates pool_en/ and hn/ subdirs')
    p.add_argument('--extract-concurrency', type=int, default=2)
    p.add_argument('--reason-concurrency', type=int, default=4)
    p.add_argument('--extract-max-tokens', type=int, default=1024)
    p.add_argument('--reason-max-tokens', type=int, default=1024)
    p.add_argument('--timeout', type=float, default=600.0)
    return p.parse_args()

def _append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    line = json.dumps(row) + '\n'
    with lock:
        with path.open('a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

def load_pool_en_candidate_index() -> tuple[list[str], np.ndarray]:
    idx_path = bsp_candidate_embeddings_index_csv()
    emb_path = bsp_candidate_embeddings_npy()
    user_ids: list[str] = []
    with idx_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            user_ids.append(row['user_id'])
    matrix = np.load(emb_path)
    if matrix.shape[0] != len(user_ids):
        raise SystemExit(f'pool_en embedding rows {matrix.shape[0]} != index {len(user_ids)}')
    return (user_ids, matrix)

def load_or_build_hn_candidate_index(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / 'hn_candidate_embeddings_mpnet.npy'
    idx_path = cache_dir / 'hn_candidate_embeddings_index.csv'
    if npy_path.exists() and idx_path.exists():
        user_ids: list[str] = []
        with idx_path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                user_ids.append(row['user_id'])
        matrix = np.load(npy_path)
        if matrix.shape[0] == len(user_ids):
            print(f'HN candidate index cache hit ({len(user_ids)} rows)')
            return (user_ids, matrix)
        print('HN candidate cache mismatch; rebuilding')
    cand_dir = bsp_hn_candidate_summaries()
    paths = sorted(cand_dir.glob('user_*_summary.json'))
    if not paths:
        raise SystemExit(f'No HN candidate summaries in {cand_dir}')
    user_ids = []
    texts = []
    for p in paths:
        stem = p.stem
        uid = stem[:-len('_summary')] if stem.endswith('_summary') else stem
        user_ids.append(uid)
        texts.append(load_hn_candidate_summary_text(uid))
    print(f'Embedding {len(texts)} HN candidate summaries with all-mpnet…')
    matrix = embed_texts(texts)
    np.save(npy_path, matrix)
    with idx_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['row_id', 'user_id'])
        w.writeheader()
        for (i, uid) in enumerate(user_ids):
            w.writerow({'row_id': i, 'user_id': uid})
    print(f'Wrote HN candidate cache → {npy_path}')
    return (user_ids, matrix)

def phase_extract(users: list[str], *, pool: str, level: str, extract_model: str, extract_tpl: str, merge_tpl: str, out_dir: Path, resume: bool, max_tokens: int, concurrency: int, timeout: float, load_profile: Callable[[str], str]) -> dict[str, dict]:
    extract_dir = out_dir / 'extract'
    extract_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'extract_meta.jsonl'
    done = load_resume_skip_user_ids(meta_path) if resume else set()
    pending = [u for u in users if u not in done]
    n_ok = len(load_ok_user_ids(meta_path)) if resume else 0
    print(f'[{pool}] extract: {n_ok} ok cached, {len(done) - n_ok} permanent-skip, {len(pending)} pending (K={concurrency})')
    if not pending:
        return load_last_ok_by_user(meta_path)
    write_lock = threading.Lock()

    def _one(uid: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        row: dict[str, Any] = {'user_id': uid, 'query_user_id': uid, 'pool': pool, 'T_level': level, 'status': 'error', 'error': '', 'error_type': ''}
        try:
            text = load_profile(uid)
            row['n_words'] = len(text.split())
            client = get_client(timeout=timeout, max_retries=0)
            result = extract_profile(text, client=client, model=extract_model, extract_template=extract_tpl, merge_template=merge_tpl, max_tokens=max_tokens, enable_thinking=False, seed=None)
            summary_path = extract_dir / f'{uid}.txt'
            summary_path.write_text(result.summary + '\n', encoding='utf-8')
            row.update({'status': 'ok', 'chunked': result.chunked, 'n_chunks': result.n_chunks, 'method': result.method, 'chunk_sizes': list(result.chunk_sizes), 'probe_tokens_full': result.probe_tokens_full, 'probe_tokens_per_chunk': list(result.probe_tokens_per_chunk), 'summary_chars': len(result.summary)})
        except (ContextLengthExceededError, PermanentRequestError) as exc:
            row['status'] = 'permanent_error'
            row['error_type'] = type(exc).__name__
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['traceback_tail'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['traceback_tail'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        row['latency_s'] = round(time.perf_counter() - t0, 3)
        _append_jsonl(meta_path, row, write_lock)
        status = row['status']
        print(f"[{pool}] extract {status} {uid} chunked={row.get('chunked')} n_chunks={row.get('n_chunks')} {row['latency_s']:.1f}s", flush=True)
        return row
    if concurrency <= 1:
        for uid in pending:
            _one(uid)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool_ex:
            futs = [pool_ex.submit(_one, uid) for uid in pending]
            for fut in as_completed(futs):
                fut.result()
    return load_last_ok_by_user(meta_path)

def phase_search(users: list[str], *, pool: str, extract_dir: Path, out_dir: Path, cand_ids: list[str], cand_matrix: np.ndarray) -> Path:
    summaries: list[str] = []
    ok_users: list[str] = []
    for uid in users:
        sp = extract_dir / f'{uid}.txt'
        if not sp.exists():
            print(f'[{pool}] WARN skip search {uid}: no summary')
            continue
        summaries.append(sp.read_text(encoding='utf-8').strip())
        ok_users.append(uid)
    if not ok_users:
        raise SystemExit(f'[{pool}] No summaries for search phase')
    print(f'[{pool}] search: embedding {len(ok_users)} query summaries…')
    q_vecs = embed_texts(summaries)
    hits = search_top_k(q_vecs, cand_matrix, cand_ids, k=15)
    search_path = out_dir / 'search_top15.csv'
    tmp_path = out_dir / 'search_top15.csv.tmp'
    with tmp_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['query_user_id', 'rank', 'candidate_user_id', 'score'])
        w.writeheader()
        for (uid, user_hits) in zip(ok_users, hits):
            for h in user_hits:
                w.writerow({'query_user_id': uid, 'rank': h.rank, 'candidate_user_id': h.candidate_user_id, 'score': f'{h.score:.6f}'})
    tmp_path.replace(search_path)
    print(f'[{pool}] search wrote {search_path} ({len(ok_users)} queries)')
    return search_path

def phase_reason(users: list[str], *, pool: str, level: str, extract_dir: Path, search_path: Path, reason_model: str, reason_tpl: str, out_dir: Path, seed: int, resume: bool, max_tokens: int, concurrency: int, timeout: float, load_cand_summary: Callable[[str], str]) -> Path:
    by_query: dict[str, list[dict]] = {u: [] for u in users}
    with search_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by_query.setdefault(row['query_user_id'], []).append(row)
    out_jsonl = out_dir / 'reason_predictions.jsonl'
    done = load_resume_skip_user_ids(out_jsonl) if resume else set()
    pending = [u for u in users if u not in done]
    n_ok = len(load_ok_user_ids(out_jsonl)) if resume else 0
    print(f'[{pool}] reason: {n_ok} ok cached, {len(done) - n_ok} permanent-skip, {len(pending)} pending (K={concurrency})')
    if not pending:
        return out_jsonl
    write_lock = threading.Lock()

    def _one(uid: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        rows = sorted(by_query.get(uid, []), key=lambda r: int(r['rank']))
        row: dict[str, Any] = {'pool': pool, 'T_level': level, 'query_user_id': uid, 'status': 'error', 'error': '', 'error_type': '', 'model': reason_model, 'max_tokens': max_tokens}
        try:
            if len(rows) < 15:
                print(f'[{pool}] WARN {uid}: only {len(rows)} search hits')
            qsum_path = extract_dir / f'{uid}.txt'
            if not qsum_path.exists():
                raise FileNotFoundError(f'Missing extract summary: {qsum_path}')
            q_text = qsum_path.read_text(encoding='utf-8').strip()
            cands = []
            for r in rows[:15]:
                cid = r['candidate_user_id']
                cands.append({'candidate_user_id': cid, 'rank': int(r['rank']), 'score': float(r['score']), 'summary': load_cand_summary(cid)})
            block = build_candidate_block(cands)
            user_prompt = build_user_prompt(reason_tpl, q_text, block)
            client = get_client(timeout=timeout, max_retries=0)
            result = generate([{'role': 'user', 'content': user_prompt}], model=reason_model, client=client, temperature=0.0, max_tokens=max_tokens, seed=seed, enable_thinking=False)
            obj = extract_json_object_for_reason(result.text, reason_model=reason_model)
            (pred_id, pred_num, err) = resolve_predicted_candidate_id(obj, [c['candidate_user_id'] for c in cands])
            if err:
                raise ValueError(err)
            row.update({'status': 'ok', 'selected_candidate_user_id': pred_id, 'selected_candidate_number': pred_num, 'correct': pred_id == uid, 'true_in_top15': any((c['candidate_user_id'] == uid for c in cands))})
        except (ContextLengthExceededError, PermanentRequestError) as exc:
            row['status'] = 'permanent_error'
            row['error_type'] = type(exc).__name__
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['traceback_tail'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['traceback_tail'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            if isinstance(exc, JsonParseError) and exc.raw_text:
                row['raw_text'] = exc.raw_text[:8000]
        row['latency_s'] = round(time.perf_counter() - t0, 3)
        _append_jsonl(out_jsonl, row, write_lock)
        print(f"[{pool}] reason {row['status']} {uid} pick={row.get('selected_candidate_number')} correct={row.get('correct')} {row['latency_s']:.1f}s", flush=True)
        return row
    if concurrency <= 1:
        for uid in pending:
            _one(uid)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool_ex:
            futs = [pool_ex.submit(_one, uid) for uid in pending]
            for fut in as_completed(futs):
                fut.result()
    return out_jsonl

def evaluate(reason_jsonl: Path, extract_meta: dict[str, dict]) -> dict[str, Any]:
    last_ok = load_last_ok_by_user(reason_jsonl)
    ok = list(last_ok.values())
    n = len(ok)
    top1 = sum((1 for p in ok if p.get('correct')))
    hit15 = sum((1 for p in ok if p.get('true_in_top15')))
    chunked = sum((1 for m in extract_meta.values() if m.get('chunked')))
    return {'n_reason_ok': n, 'top1_correct': top1, 'top1_accuracy': top1 / n if n else 0.0, 'hit_at_15_count': hit15, 'hit_at_15': hit15 / n if n else 0.0, 'n_chunked_extract': chunked, 'n_single_extract': len(extract_meta) - chunked}

def run_pool(pool: str, *, args: argparse.Namespace, parent_out: Path, extract_tpl: str, merge_tpl: str, reason_tpl: str) -> Path:
    out_dir = parent_out / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    if pool == 'pool_en':
        users = list_pool_en_query_user_ids(args.level)
        load_profile = lambda uid: bsp_truncated_query(args.level, uid).read_text(encoding='utf-8')
        cand_summ_dir = bsp_pool_en_candidate_summaries()

        def load_cand_summary(cid: str) -> str:
            return (cand_summ_dir / f'{cid}.txt').read_text(encoding='utf-8').strip()
        cand_ids: Optional[list[str]] = None
        cand_matrix: Optional[np.ndarray] = None
    elif pool == 'hn':
        users = list_hn_query_user_ids(args.level)
        load_profile = lambda uid: load_hn_query_profile_text(uid, args.level)
        load_cand_summary = load_hn_candidate_summary_text
        cand_ids = None
        cand_matrix = None
    else:
        raise SystemExit(f'Unknown pool: {pool}')
    if args.user_ids:
        want = {u.strip() for u in args.user_ids.split(',') if u.strip()}
        users = [u for u in users if u in want]
        missing = sorted(want - set(users))
        if missing:
            print(f'[{pool}] WARN unknown/missing user_ids: {missing}', flush=True)
    if args.limit is not None:
        users = users[:args.limit]
    print(f'[{pool}] n_users={len(users)} level={args.level} out={out_dir}')
    extract_meta: dict[str, dict] = {}
    if args.phase in ('all', 'extract'):
        extract_meta = phase_extract(users, pool=pool, level=args.level, extract_model=args.extract_model, extract_tpl=extract_tpl, merge_tpl=merge_tpl, out_dir=out_dir, resume=args.resume, max_tokens=args.extract_max_tokens, concurrency=args.extract_concurrency, timeout=args.timeout, load_profile=load_profile)
    else:
        extract_meta = load_last_ok_by_user(out_dir / 'extract_meta.jsonl')
    search_path = out_dir / 'search_top15.csv'
    if args.phase in ('all', 'search'):
        if pool == 'pool_en':
            (cand_ids, cand_matrix) = load_pool_en_candidate_index()
        else:
            (cand_ids, cand_matrix) = load_or_build_hn_candidate_index(out_dir / 'cache')
        assert cand_ids is not None and cand_matrix is not None
        search_users = [u for u in users if (out_dir / 'extract' / f'{u}.txt').exists()]
        phase_search(search_users, pool=pool, extract_dir=out_dir / 'extract', out_dir=out_dir, cand_ids=cand_ids, cand_matrix=cand_matrix)
    reason_jsonl = out_dir / 'reason_predictions.jsonl'
    if args.phase in ('all', 'reason'):
        if not search_path.exists():
            raise SystemExit(f'[{pool}] Missing {search_path}; run --phase search first')
        search_users = [u for u in users if (out_dir / 'extract' / f'{u}.txt').exists()]
        phase_reason(search_users, pool=pool, level=args.level, extract_dir=out_dir / 'extract', search_path=search_path, reason_model=args.reason_model, reason_tpl=reason_tpl, out_dir=out_dir, seed=args.seed, resume=args.resume, max_tokens=args.reason_max_tokens, concurrency=args.reason_concurrency, timeout=args.timeout, load_cand_summary=load_cand_summary)
    metrics: dict[str, Any] = {}
    if reason_jsonl.exists():
        metrics = evaluate(reason_jsonl, extract_meta)
        (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
        print(f"[{pool}] metrics top1={metrics['top1_correct']}/{metrics['n_reason_ok']} hit@15={metrics['hit_at_15_count']}/{metrics['n_reason_ok']} chunked={metrics['n_chunked_extract']}")
    write_manifest(out_dir / 'manifest.json', {'task': 'p1_full_pool', 'pool': pool, 'T_level': args.level, 'extract_model': args.extract_model, 'reason_model': args.reason_model, 'phase': args.phase, 'n_users': len(users), 'extract_concurrency': args.extract_concurrency, 'reason_concurrency': args.reason_concurrency, 'resume': args.resume, 'metrics': metrics, 'git_commit': git_commit(summer_root()), 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    return out_dir

def preflight(pools: list[str], level: str) -> None:
    missing: list[str] = []
    for (label, path) in [('summarization prompt', bsp_prompt_summarization()), ('record_selection prompt', bsp_prompt_record_selection()), ('extract_merge prompt', summer_prompt_extract_merge())]:
        if not path.exists():
            missing.append(f'{label}: {path}')
    if 'pool_en' in pools:
        checks = [bsp_pool_en() / 'truncated_queries' / level, bsp_pool_en_candidate_summaries(), bsp_candidate_embeddings_npy(), bsp_candidate_embeddings_index_csv()]
        for path in checks:
            if not path.exists():
                missing.append(f'pool_en: {path}')
        n_q = len(list_pool_en_query_user_ids(level)) if not missing else 0
        print(f'preflight pool_en: truncated_queries/{level} n={n_q}')
    if 'hn' in pools:
        checks = [bsp_hn_truncated_dir(level), bsp_hn_candidate_summaries(), bsp_hn_user_mapping(), bsp_hn_pool_manifest()]
        for path in checks:
            if not path.exists():
                missing.append(f'hn: {path}')
        if not missing or all((not m.startswith('hn:') for m in missing)):
            n_q = len(list_hn_query_user_ids(level))
            n_c = len(list(bsp_hn_candidate_summaries().glob('user_*_summary.json')))
            print(f'preflight hn: query={n_q} candidate_summaries={n_c}')
    if missing:
        raise SystemExit('Preflight failed:\n  - ' + '\n  - '.join(missing))

def main() -> int:
    load_dotenv_files(summer_root())
    load_dotenv_files(summer_root().parent)
    args = parse_args()
    if args.extract_concurrency < 1 or args.reason_concurrency < 1:
        raise SystemExit('concurrency must be >= 1')
    pools = ['pool_en', 'hn'] if args.pool == 'both' else [args.pool]
    preflight(pools, args.level)
    parent_out = Path(args.out_dir) if args.out_dir else summer_root() / 'results' / 'runs' / f"p1_full_pool_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    parent_out.mkdir(parents=True, exist_ok=True)
    extract_tpl = load_prompt_template(bsp_prompt_summarization())
    merge_tpl = load_prompt_template(summer_prompt_extract_merge())
    reason_tpl = load_reason_template(bsp_prompt_record_selection())
    for pool in pools:
        run_pool(pool, args=args, parent_out=parent_out, extract_tpl=extract_tpl, merge_tpl=merge_tpl, reason_tpl=reason_tpl)
    write_manifest(parent_out / 'manifest.json', {'task': 'p1_full_pool_parent', 'pools': pools, 'T_level': args.level, 'extract_model': args.extract_model, 'reason_model': args.reason_model, 'extract_concurrency': args.extract_concurrency, 'reason_concurrency': args.reason_concurrency, 'resume': args.resume, 'phase': args.phase, 'created_at_utc': datetime.now(timezone.utc).isoformat()})
    print(f'Done → {parent_out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
