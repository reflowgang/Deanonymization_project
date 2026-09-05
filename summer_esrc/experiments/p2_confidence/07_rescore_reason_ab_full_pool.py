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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.confidence import estimator_a_verbalized, estimator_b_selected_id_logprob, logprob_to_unit_interval
from esrc.config import load_dotenv_files
from esrc.generate import ContextLengthExceededError, PermanentRequestError, generate, get_client
from esrc.paths import bsp_pool_en_candidate_summaries, bsp_prompt_record_selection, load_hn_candidate_summary_text, summer_root
from esrc.reason_prompt import build_candidate_block, build_user_prompt, clamp01, extract_json_object_for_reason, JsonParseError, load_prompt_template, resolve_predicted_candidate_id
from esrc.resume import load_ok_user_ids, load_resume_skip_user_ids
DEFAULT_REASON_MODEL = 'qwen3.6-35b-a3b-nvfp4'
DEFAULT_P1_RUN = summer_root() / 'results' / 'runs' / 'p1_full_pool_overnight_20260805'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P2 (a)/(b) Reason re-score on full pools')
    p.add_argument('--p1-run-dir', type=str, default=str(DEFAULT_P1_RUN))
    p.add_argument('--out-dir', type=str, default=str(summer_root() / 'results' / 'runs' / 'p2_full_pool_ab_rescore'))
    p.add_argument('--pool', choices=('pool_en', 'hn', 'both'), default='both')
    p.add_argument('--model', default=os.getenv('VLLM_REASON_MODEL', DEFAULT_REASON_MODEL))
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--user-ids', type=str, default=None, help='Comma-separated query_user_ids to run (smoke / subset). Applied within each selected --pool.')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()

def _append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False) + '\n'
    with lock:
        with path.open('a', encoding='utf-8') as f:
            f.write(line)

def _append_scores_csv(path: Path, row: dict, fieldnames: list[str], lock: threading.Lock) -> None:
    with lock:
        new_file = not path.exists()
        with path.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in fieldnames})
SCORE_FIELDS = ['pool', 'T_level', 'query_user_id', 'selected_candidate_user_id', 'selected_candidate_number', 'correct', 'true_in_top15', 'score_a', 'score_b', 'verbalized_confidence', 'selected_id_logprob', 'status', 'error', 'model', 'latency_s']

def run_pool(pool: str, *, p1_dir: Path, out_dir: Path, model: str, seed: int, max_tokens: int, timeout: float, concurrency: int, resume: bool, limit: Optional[int], user_ids: Optional[set[str]], reason_tpl: str, load_cand: Callable[[str], str]) -> None:
    extract_dir = p1_dir / 'extract'
    search_path = p1_dir / 'search_top15.csv'
    if not search_path.exists():
        raise SystemExit(f'Missing {search_path}')
    by_query: dict[str, list[dict]] = {}
    with search_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by_query.setdefault(row['query_user_id'], []).append(row)
    users = sorted((u for u in by_query if (extract_dir / f'{u}.txt').exists()))
    if user_ids is not None:
        users = [u for u in users if u in user_ids]
        missing = sorted(user_ids - set(users))
        if missing:
            print(f'[{pool}] WARN unknown/missing user_ids: {missing}', flush=True)
    if limit is not None:
        users = users[:limit]
    pool_out = out_dir / pool
    pool_out.mkdir(parents=True, exist_ok=True)
    out_jsonl = pool_out / 'reason_predictions.jsonl'
    scores_csv = pool_out / 'estimator_scores.csv'
    done = load_resume_skip_user_ids(out_jsonl) if resume else set()
    if resume and scores_csv.exists():
        with scores_csv.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r.get('status') == 'ok' and r.get('query_user_id'):
                    done.add(r['query_user_id'])
    pending = [u for u in users if u not in done]
    n_ok_cached = len(load_ok_user_ids(out_jsonl)) if out_jsonl.exists() else 0
    print(f'[{pool}] reason-ab: {n_ok_cached} ok cached, {len(done) - n_ok_cached} permanent-skip, {len(pending)} pending (K={concurrency})', flush=True)
    if not pending:
        return
    write_lock = threading.Lock()

    def _one(uid: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        rows = sorted(by_query.get(uid, []), key=lambda r: int(r['rank']))
        row: dict[str, Any] = {'pool': pool, 'T_level': 'T8', 'query_user_id': uid, 'status': 'error', 'error': '', 'error_type': '', 'model': model, 'enable_thinking': False, 'max_tokens': max_tokens}
        score_row: dict[str, Any] = {'pool': pool, 'T_level': 'T8', 'query_user_id': uid, 'status': 'error', 'error': '', 'model': model}
        try:
            q_text = (extract_dir / f'{uid}.txt').read_text(encoding='utf-8').strip()
            cands = []
            for r in rows[:15]:
                cid = r['candidate_user_id']
                cands.append({'candidate_user_id': cid, 'rank': int(r['rank']), 'score': float(r['score']), 'summary': load_cand(cid)})
            cand_ids = [c['candidate_user_id'] for c in cands]
            block = build_candidate_block(cands)
            user_prompt = build_user_prompt(reason_tpl, q_text, block)
            client = get_client(timeout=timeout, max_retries=0)
            result = generate([{'role': 'user', 'content': user_prompt}], model=model, client=client, temperature=0.0, max_tokens=max_tokens, logprobs=True, top_logprobs=5, seed=seed, enable_thinking=False)
            if not result.text and result.reasoning_content:
                raise ValueError('Empty content with non-empty reasoning_content')
            obj = extract_json_object_for_reason(result.text, reason_model=model)
            (pred_id, pred_num, err) = resolve_predicted_candidate_id(obj, cand_ids)
            if err:
                raise ValueError(err)
            conf = clamp01(obj.get('confidence', 0.0))
            id_lp = estimator_b_selected_id_logprob(result.token_logprobs, pred_num)
            score_a = estimator_a_verbalized(conf)
            score_b = logprob_to_unit_interval(id_lp) if id_lp is not None else float('nan')
            correct = pred_id == uid
            true_in = any((c['candidate_user_id'] == uid for c in cands))
            row.update({'status': 'ok', 'selected_candidate_user_id': pred_id, 'selected_candidate_number': pred_num, 'verbalized_confidence': conf, 'reasoning_short': str(obj.get('reasoning_short', ''))[:500], 'correct': correct, 'true_in_top15': true_in, 'candidate_user_ids': cand_ids, 'sequence_logprob_full': result.sequence_logprob, 'selected_id_logprob': id_lp, 'n_token_logprobs': len(result.token_logprobs), 'raw_text': result.text, 'finish_reason': result.finish_reason, 'token_logprobs': [{'token': t.token, 'logprob': t.logprob} for t in result.token_logprobs]})
            score_row.update({'status': 'ok', 'selected_candidate_user_id': pred_id, 'selected_candidate_number': pred_num, 'correct': correct, 'true_in_top15': true_in, 'score_a': score_a, 'score_b': score_b, 'verbalized_confidence': conf, 'selected_id_logprob': id_lp})
        except (ContextLengthExceededError, PermanentRequestError) as exc:
            row['status'] = 'permanent_error'
            row['error_type'] = type(exc).__name__
            row['error'] = f'{type(exc).__name__}: {exc}'
            score_row['status'] = 'permanent_error'
            score_row['error'] = row['error']
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['traceback_tail'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            if isinstance(exc, JsonParseError) and exc.raw_text:
                row['raw_text'] = exc.raw_text[:8000]
            score_row['error'] = row['error']
        lat = round(time.perf_counter() - t0, 3)
        row['latency_s'] = lat
        score_row['latency_s'] = lat
        _append_jsonl(out_jsonl, row, write_lock)
        _append_scores_csv(scores_csv, score_row, SCORE_FIELDS, write_lock)
        print(f"[{pool}] reason-ab {row['status']} {uid} pick={row.get('selected_candidate_number')} conf={row.get('verbalized_confidence')} correct={row.get('correct')} {lat:.1f}s", flush=True)
        return row
    if concurrency <= 1:
        for uid in pending:
            _one(uid)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_one, uid) for uid in pending]
            for fut in as_completed(futs):
                fut.result()

def main() -> int:
    args = parse_args()
    load_dotenv_files(summer_root())
    p1_root = Path(args.p1_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reason_tpl = load_prompt_template(bsp_prompt_record_selection())
    pools = ['pool_en', 'hn'] if args.pool == 'both' else [args.pool]
    user_filter: Optional[set[str]] = None
    if args.user_ids:
        user_filter = {u.strip() for u in args.user_ids.split(',') if u.strip()}
    for pool in pools:
        p1_dir = p1_root / pool
        if pool == 'pool_en':
            cand_dir = bsp_pool_en_candidate_summaries()

            def load_cand(cid: str, _d=cand_dir) -> str:
                return (_d / f'{cid}.txt').read_text(encoding='utf-8').strip()
        else:
            load_cand = load_hn_candidate_summary_text
        pool_filter = user_filter
        run_pool(pool, p1_dir=p1_dir, out_dir=out_dir, model=args.model, seed=args.seed, max_tokens=args.max_tokens, timeout=args.timeout, concurrency=args.concurrency, resume=args.resume, limit=args.limit, user_ids=pool_filter, reason_tpl=reason_tpl, load_cand=load_cand)
    (out_dir / 'README.md').write_text(f'# P2 (a)/(b) Reason re-score\n\nCreated {datetime.now(timezone.utc).isoformat()}\nSource P1 extracts/search: `{p1_root}`\nReason re-run with logprobs for estimators (a)/(b). No (c).\n', encoding='utf-8')
    print(f'Done → {out_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
