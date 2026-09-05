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
from esrc.confidence import binary_choice_probability
from esrc.config import load_dotenv_files
from esrc.generate import ContextLengthExceededError, PermanentRequestError, generate, get_client
from esrc.paths import bsp_pool_en_candidate_summaries, load_hn_candidate_summary_text, summer_root
from esrc.reason_prompt import load_prompt_template
DEFAULT_MODEL = 'qwen3.6-35b-a3b-nvfp4'
DEFAULT_P1 = summer_root() / 'results' / 'runs' / 'p1_full_pool_overnight_20260805'
DEFAULT_P2 = summer_root() / 'results' / 'runs' / 'p2_full_pool_ab_rescore'
OUT_DEFAULT = summer_root() / 'results' / 'runs' / 'p2_c_redesign_full_pool'
SCORE_FIELDS = ['pool', 'query_user_id', 'correct', 'true_in_top15', 'selected_candidate_number', 'selected_candidate_user_id', 'score_a', 'score_b', 'score_c1_p_yes', 'score_c2_mean_p_a', 'c2_n_rivals', 'c2_min_p_a', 'c1_raw', 'status', 'error', 'latency_s', 'model']

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Full-pool C1/C2 redesigned (c)')
    p.add_argument('--p1-run-dir', default=str(DEFAULT_P1))
    p.add_argument('--p2-scores-dir', default=str(DEFAULT_P2))
    p.add_argument('--out-dir', default=str(OUT_DEFAULT))
    p.add_argument('--pool', choices=('pool_en', 'hn', 'both'), default='both')
    p.add_argument('--model', default=os.getenv('VLLM_REASON_MODEL', DEFAULT_MODEL))
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--max-tokens', type=int, default=8)
    p.add_argument('--top-logprobs', type=int, default=20)
    p.add_argument('--n-rivals', type=int, default=3)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--user-ids', type=str, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-c2', action='store_true')
    return p.parse_args()

def fmt(x: Optional[float], d: int=6) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return ''
    return f'{x:.{d}f}'

def load_ab_scores(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    if not path.exists():
        return by
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            uid = r.get('query_user_id')
            if uid and r.get('status') == 'ok':
                by[uid] = r
    return by

def load_ok_reason(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        uid = o.get('query_user_id')
        if uid and o.get('status') == 'ok':
            by[uid] = o
    return by

def load_search(path: Path) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            by.setdefault(r['query_user_id'], []).append(r)
    for uid in by:
        by[uid] = sorted(by[uid], key=lambda r: int(r['rank']))
    return by

def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status') in ('ok', 'permanent_error') and r.get('query_user_id'):
                done.add(r['query_user_id'])
    return done

def build_pack(uid: str, *, extract_dir: Path, search_rows: list[dict], load_cand: Callable[[str], str], reason: dict) -> dict:
    q = (extract_dir / f'{uid}.txt').read_text(encoding='utf-8').strip()
    cands = []
    for r in search_rows[:15]:
        cid = r['candidate_user_id']
        cands.append({'candidate_user_id': cid, 'rank': int(r['rank']), 'score': float(r['score']), 'summary': load_cand(cid)})
    return {'query_user_id': uid, 'query_summary': q, 'candidates': cands, 'selected_candidate_number': int(reason['selected_candidate_number']), 'selected_candidate_user_id': reason.get('selected_candidate_user_id'), 'correct': bool(reason.get('correct'))}

def cand_by_number(pack: dict, number: int) -> dict:
    for c in pack['candidates']:
        if int(c['rank']) == int(number):
            return c
    raise KeyError(f'candidate number {number} missing')

def rivals_for_c2(pack: dict, pick_num: int, n_rivals: int) -> list[dict]:
    ordered = sorted(pack['candidates'], key=lambda c: int(c['rank']))
    return [c for c in ordered if int(c['rank']) != int(pick_num)][:n_rivals]

def one_shot_binary(*, client, model: str, prompt: str, positive: str, negative: str, max_tokens: int, top_logprobs: int, seed: int) -> dict[str, Any]:
    result = generate([{'role': 'user', 'content': prompt}], model=model, client=client, temperature=0.0, max_tokens=max_tokens, logprobs=True, top_logprobs=top_logprobs, seed=seed, enable_thinking=False)
    parsed = binary_choice_probability(result.token_logprobs, positive, negative)
    out: dict[str, Any] = {'raw_text': (result.text or '')[:80], 'finish_reason': result.finish_reason}
    if parsed is None:
        out['ok'] = False
        out['error'] = f'missing {positive}/{negative} in first-token top_logprobs'
        return out
    (p_pos, lp_pos, lp_neg) = parsed
    out.update({'ok': True, 'p_positive': p_pos, 'lp_positive': lp_pos, 'lp_negative': lp_neg})
    return out

def run_pool(pool: str, *, p1_dir: Path, p2_dir: Path, out_dir: Path, args: argparse.Namespace, tpl_c1: str, tpl_c2: str, load_cand: Callable[[str], str], user_filter: Optional[set[str]]) -> None:
    reason_by = load_ok_reason(p2_dir / 'reason_predictions.jsonl')
    ab_by = load_ab_scores(p2_dir / 'estimator_scores.csv')
    search_by = load_search(p1_dir / 'search_top15.csv')
    extract_dir = p1_dir / 'extract'
    users = sorted((uid for uid in reason_by if uid in ab_by and uid in search_by and (extract_dir / f'{uid}.txt').exists()))
    if user_filter is not None:
        users = [u for u in users if u in user_filter]
        missing = sorted(user_filter - set(users))
        if missing:
            print(f'[{pool}] WARN unknown/missing user_ids: {missing}', flush=True)
    if args.limit is not None:
        users = users[:args.limit]
    pool_out = out_dir / pool
    pool_out.mkdir(parents=True, exist_ok=True)
    scores_path = pool_out / 'c_redesign_scores.csv'
    detail_path = pool_out / 'c_redesign_details.jsonl'
    done = load_done(scores_path) if args.resume else set()
    pending = [u for u in users if u not in done]
    print(f'[{pool}] C1/C2: {len(done)} skip, {len(pending)} pending (K={args.concurrency}, rivals={args.n_rivals})', flush=True)
    if not pending:
        return
    write_lock = threading.Lock()

    def _one(uid: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        reason = reason_by[uid]
        ab = ab_by[uid]
        out: dict[str, Any] = {'pool': pool, 'query_user_id': uid, 'correct': str(bool(reason.get('correct'))), 'true_in_top15': str(bool(reason.get('true_in_top15'))), 'selected_candidate_number': reason.get('selected_candidate_number'), 'selected_candidate_user_id': reason.get('selected_candidate_user_id'), 'score_a': ab.get('score_a', ''), 'score_b': ab.get('score_b', ''), 'score_c1_p_yes': '', 'score_c2_mean_p_a': '', 'c2_n_rivals': args.n_rivals, 'c2_min_p_a': '', 'c1_raw': '', 'status': 'error', 'error': '', 'model': args.model}
        detail: dict[str, Any] = {'pool': pool, 'query_user_id': uid, 'c1': None, 'c2': []}
        try:
            pack = build_pack(uid, extract_dir=extract_dir, search_rows=search_by[uid], load_cand=load_cand, reason=reason)
            pick_num = int(pack['selected_candidate_number'])
            pick = cand_by_number(pack, pick_num)
            q = pack['query_summary']
            client = get_client(timeout=args.timeout, max_retries=0)
            prompt_c1 = tpl_c1.replace('{k}', str(pick_num)).replace('{query_summary}', q).replace('{candidate_k_summary}', pick['summary'])
            c1 = one_shot_binary(client=client, model=args.model, prompt=prompt_c1, positive='YES', negative='NO', max_tokens=args.max_tokens, top_logprobs=args.top_logprobs, seed=args.seed)
            detail['c1'] = {k: c1.get(k) for k in ('ok', 'p_positive', 'raw_text', 'error')}
            if not c1.get('ok'):
                raise ValueError(c1.get('error') or 'C1 failed')
            out['score_c1_p_yes'] = fmt(c1['p_positive'])
            out['c1_raw'] = c1.get('raw_text', '')
            if not args.skip_c2:
                p_as: list[float] = []
                for rival in rivals_for_c2(pack, pick_num, args.n_rivals):
                    j = int(rival['rank'])
                    prompt_c2 = tpl_c2.replace('{k}', str(pick_num)).replace('{j}', str(j)).replace('{query_summary}', q).replace('{candidate_k_summary}', pick['summary']).replace('{candidate_j_summary}', rival['summary'])
                    c2 = one_shot_binary(client=client, model=args.model, prompt=prompt_c2, positive='A', negative='B', max_tokens=args.max_tokens, top_logprobs=args.top_logprobs, seed=args.seed + j)
                    detail['c2'].append({'rival_number': j, 'ok': c2.get('ok'), 'p_positive': c2.get('p_positive'), 'error': c2.get('error')})
                    if not c2.get('ok'):
                        raise ValueError(f"C2 vs {j}: {c2.get('error')}")
                    p_as.append(float(c2['p_positive']))
                out['score_c2_mean_p_a'] = fmt(sum(p_as) / len(p_as))
                out['c2_min_p_a'] = fmt(min(p_as))
            out['status'] = 'ok'
        except (ContextLengthExceededError, PermanentRequestError) as exc:
            out['status'] = 'permanent_error'
            out['error'] = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            out['error'] = f'{type(exc).__name__}: {exc}'
            detail['traceback'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        out['latency_s'] = f'{time.perf_counter() - t0:.3f}'
        with write_lock:
            new_file = not scores_path.exists()
            with scores_path.open('a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow({k: out.get(k, '') for k in SCORE_FIELDS})
            with detail_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(detail, ensure_ascii=False) + '\n')
        print(f"[{pool}] {out['status']} {uid} c1={out['score_c1_p_yes']} c2={out['score_c2_mean_p_a']} correct={out['correct']} {out['latency_s']}s", flush=True)
        return out
    if args.concurrency <= 1:
        for uid in pending:
            _one(uid)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_one, uid) for uid in pending]
            for fut in as_completed(futs):
                fut.result()

def main() -> int:
    load_dotenv_files()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1_root = Path(args.p1_run_dir)
    p2_root = Path(args.p2_scores_dir)
    tpl_c1 = load_prompt_template(summer_root() / 'prompts' / 'estimator_c1_confirm.txt')
    tpl_c2 = load_prompt_template(summer_root() / 'prompts' / 'estimator_c2_pairwise.txt')
    user_filter: Optional[set[str]] = None
    if args.user_ids:
        user_filter = {u.strip() for u in args.user_ids.split(',') if u.strip()}
    pools = ['pool_en', 'hn'] if args.pool == 'both' else [args.pool]
    for pool in pools:
        if pool == 'pool_en':
            cand_dir = bsp_pool_en_candidate_summaries()

            def load_cand(cid: str, _d=cand_dir) -> str:
                return (_d / f'{cid}.txt').read_text(encoding='utf-8').strip()
        else:
            load_cand = load_hn_candidate_summary_text
        run_pool(pool, p1_dir=p1_root / pool, p2_dir=p2_root / pool, out_dir=out_dir, args=args, tpl_c1=tpl_c1, tpl_c2=tpl_c2, load_cand=load_cand, user_filter=user_filter)
    meta = {'task': 'p2_c_redesign_full_pool', 'p1_run_dir': str(p1_root), 'p2_scores_dir': str(p2_root), 'model': args.model, 'n_rivals': args.n_rivals, 'concurrency': args.concurrency, 'seed': args.seed, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (out_dir / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Done → {out_dir}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
