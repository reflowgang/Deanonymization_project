from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.confidence import binary_choice_probability
from esrc.config import load_dotenv_files
from esrc.generate import ContextLengthExceededError, PermanentRequestError, generate, get_client
from esrc.paths import summer_root
from esrc.reason_prompt import load_prompt_template
DEFAULT_MODEL = 'qwen3.6-35b-a3b-nvfp4'
DEFAULT_REASON = summer_root() / 'results' / 'runs' / 'p2_regression_50_T8' / 'reason_predictions.jsonl'
DEFAULT_INPUTS = summer_root() / 'results' / 'runs' / 'p2_inputs_regression_50_T8' / 'inputs'
DEFAULT_OLD_SCORES = summer_root() / 'results' / 'runs' / 'p2_regression_50_T8' / 'estimator_scores.csv'
OUT_DEFAULT = summer_root() / 'results' / 'runs' / 'p2_c_redesign_regression_50'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P2 redesigned (c): C1 + C2-top3')
    p.add_argument('--reason-jsonl', default=str(DEFAULT_REASON))
    p.add_argument('--inputs-dir', default=str(DEFAULT_INPUTS))
    p.add_argument('--old-scores-csv', default=str(DEFAULT_OLD_SCORES))
    p.add_argument('--out-dir', default=str(OUT_DEFAULT))
    p.add_argument('--model', default=os.getenv('VLLM_REASON_MODEL', DEFAULT_MODEL))
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--timeout', type=float, default=120.0)
    p.add_argument('--max-tokens', type=int, default=8)
    p.add_argument('--top-logprobs', type=int, default=20)
    p.add_argument('--n-rivals', type=int, default=3, help='C2 rivals (Search rank order)')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--skip-c2', action='store_true')
    return p.parse_args()

def fmt(x: Optional[float], d: int=6) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and x != x:
        return ''
    return f'{x:.{d}f}'

def load_ok_reason(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get('status') == 'ok':
            rows.append(o)
    rows.sort(key=lambda r: r['query_user_id'])
    return rows

def load_pack(inputs_dir: Path, uid: str) -> dict:
    return json.loads((inputs_dir / f'{uid}.json').read_text(encoding='utf-8'))

def load_old_scores(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    if not path.exists():
        return by
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status') == 'ok' and r.get('query_user_id'):
                by[r['query_user_id']] = r
    return by

def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status') == 'ok' and r.get('query_user_id'):
                done.add(r['query_user_id'])
    return done

def cand_by_number(pack: dict, number: int) -> dict:
    for c in pack['candidates']:
        if int(c['rank']) == int(number):
            return c
    raise KeyError(f'candidate number {number} missing')

def rivals_for_c2(pack: dict, pick_num: int, n_rivals: int) -> list[dict]:
    ordered = sorted(pack['candidates'], key=lambda c: int(c['rank']))
    rivals = [c for c in ordered if int(c['rank']) != int(pick_num)]
    return rivals[:n_rivals]

def one_shot_binary(*, client, model: str, prompt: str, positive: str, negative: str, max_tokens: int, top_logprobs: int, seed: int) -> dict[str, Any]:
    result = generate([{'role': 'user', 'content': prompt}], model=model, client=client, temperature=0.0, max_tokens=max_tokens, logprobs=True, top_logprobs=top_logprobs, seed=seed, enable_thinking=False)
    parsed = binary_choice_probability(result.token_logprobs, positive, negative)
    out: dict[str, Any] = {'raw_text': (result.text or '')[:80], 'finish_reason': result.finish_reason, 'n_token_logprobs': len(result.token_logprobs), 'first_token': result.token_logprobs[0].token if result.token_logprobs else '', 'first_token_top': [{'token': t, 'logprob': lp} for (t, lp) in (result.token_logprobs[0].top_logprobs if result.token_logprobs else ())][:10]}
    if parsed is None:
        out['ok'] = False
        out['error'] = f'missing {positive}/{negative} in first-token top_logprobs'
        return out
    (p_pos, lp_pos, lp_neg) = parsed
    out.update({'ok': True, 'p_positive': p_pos, 'lp_positive': lp_pos, 'lp_negative': lp_neg})
    return out

def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum(((x - mx) * (y - my) for (x, y) in zip(xs, ys)))
    dx = math.sqrt(sum(((x - mx) ** 2 for x in xs)))
    dy = math.sqrt(sum(((y - my) ** 2 for y in ys)))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)

def main() -> int:
    load_dotenv_files()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / 'c_redesign_scores.csv'
    detail_path = out_dir / 'c_redesign_details.jsonl'
    reason_rows = load_ok_reason(Path(args.reason_jsonl))
    if args.limit is not None:
        reason_rows = reason_rows[:args.limit]
    old = load_old_scores(Path(args.old_scores_csv))
    done = load_done(scores_path) if args.resume else set()
    tpl_c1 = load_prompt_template(summer_root() / 'prompts' / 'estimator_c1_confirm.txt')
    tpl_c2 = load_prompt_template(summer_root() / 'prompts' / 'estimator_c2_pairwise.txt')
    fieldnames = ['query_user_id', 'correct', 'selected_candidate_number', 'selected_candidate_user_id', 'score_a', 'score_b', 'score_c_old', 'score_c1_p_yes', 'score_c2_mean_p_a', 'c2_n_rivals', 'c2_min_p_a', 'c1_raw', 'status', 'error', 'latency_s', 'model']
    pending = [r for r in reason_rows if r['query_user_id'] not in done]
    print(f'C redesign: {len(done)} done, {len(pending)} pending (n_rivals={args.n_rivals}, skip_c2={args.skip_c2})', flush=True)
    client = get_client(timeout=args.timeout, max_retries=0)
    inputs_dir = Path(args.inputs_dir)
    for row in pending:
        uid = row['query_user_id']
        t0 = time.perf_counter()
        out: dict[str, Any] = {'query_user_id': uid, 'correct': str(bool(row.get('correct'))), 'selected_candidate_number': row.get('selected_candidate_number'), 'selected_candidate_user_id': row.get('selected_candidate_user_id'), 'score_a': old.get(uid, {}).get('score_a', ''), 'score_b': old.get(uid, {}).get('score_b', ''), 'score_c_old': old.get(uid, {}).get('score_c', ''), 'score_c1_p_yes': '', 'score_c2_mean_p_a': '', 'c2_n_rivals': args.n_rivals, 'c2_min_p_a': '', 'c1_raw': '', 'status': 'error', 'error': '', 'model': args.model}
        detail: dict[str, Any] = {'query_user_id': uid, 'c1': None, 'c2': []}
        try:
            pack = load_pack(inputs_dir, uid)
            pick_num = int(row['selected_candidate_number'])
            pick = cand_by_number(pack, pick_num)
            q = pack['query_summary']
            prompt_c1 = tpl_c1.replace('{k}', str(pick_num)).replace('{query_summary}', q).replace('{candidate_k_summary}', pick['summary'])
            c1 = one_shot_binary(client=client, model=args.model, prompt=prompt_c1, positive='YES', negative='NO', max_tokens=args.max_tokens, top_logprobs=args.top_logprobs, seed=args.seed)
            detail['c1'] = c1
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
                    c2['rival_number'] = j
                    c2['rival_user_id'] = rival['candidate_user_id']
                    detail['c2'].append(c2)
                    if not c2.get('ok'):
                        raise ValueError(f"C2 vs rival {j}: {c2.get('error') or 'failed'}")
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
        new_file = not scores_path.exists()
        with scores_path.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            w.writerow({k: out.get(k, '') for k in fieldnames})
        with detail_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(detail, ensure_ascii=False) + '\n')
        print(f"{out['status']} {uid} c1={out['score_c1_p_yes']} c2={out['score_c2_mean_p_a']} correct={out['correct']} {out['latency_s']}s", flush=True)
    rows = list(csv.DictReader(scores_path.open(newline='', encoding='utf-8')))
    ok = [r for r in rows if r.get('status') == 'ok']
    if ok:
        y = [1.0 if str(r['correct']).lower() == 'true' else 0.0 for r in ok]

        def _col(name: str) -> list[float]:
            return [float(r[name]) for r in ok if r.get(name) not in ('', None)]
        summary = {'n_ok': len(ok), 'n_correct': int(sum(y)), 'c1_mean': sum(_col('score_c1_p_yes')) / max(1, len(_col('score_c1_p_yes'))), 'c1_std': None, 'c2_mean': None, 'corr_c1_correct': pearson(_col('score_c1_p_yes'), y[:len(_col('score_c1_p_yes'))]), 'corr_c2_correct': None, 'corr_b_correct': None, 'corr_c_old_correct': None, 'c1_mean_correct': None, 'c1_mean_incorrect': None}
        c1 = _col('score_c1_p_yes')
        if len(c1) == len(y):
            mc = sum(c1) / len(c1)
            summary['c1_mean'] = mc
            summary['c1_std'] = math.sqrt(sum(((x - mc) ** 2 for x in c1)) / len(c1))
            summary['c1_mean_correct'] = sum((x for (x, yy) in zip(c1, y) if yy == 1.0)) / max(1, sum((1 for yy in y if yy == 1.0)))
            summary['c1_mean_incorrect'] = sum((x for (x, yy) in zip(c1, y) if yy == 0.0)) / max(1, sum((1 for yy in y if yy == 0.0)))
            summary['corr_c1_correct'] = pearson(c1, y)
        c2 = _col('score_c2_mean_p_a')
        if len(c2) == len(y) and c2:
            summary['c2_mean'] = sum(c2) / len(c2)
            summary['c2_mean_correct'] = sum((x for (x, yy) in zip(c2, y) if yy == 1.0)) / max(1, sum((1 for yy in y if yy == 1.0)))
            summary['c2_mean_incorrect'] = sum((x for (x, yy) in zip(c2, y) if yy == 0.0)) / max(1, sum((1 for yy in y if yy == 0.0)))
            summary['corr_c2_correct'] = pearson(c2, y)
        b = _col('score_b')
        if len(b) == len(y):
            summary['corr_b_correct'] = pearson(b, y)
        cold = _col('score_c_old')
        if len(cold) == len(y):
            summary['corr_c_old_correct'] = pearson(cold, y)
            summary['c_old_mean'] = sum(cold) / len(cold)
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print('summary:', json.dumps(summary, indent=2), flush=True)
    meta = {'task': 'p2_c_redesign_c1_c2', 'reason_jsonl': args.reason_jsonl, 'inputs_dir': args.inputs_dir, 'model': args.model, 'n_rivals': args.n_rivals, 'seed': args.seed, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (out_dir / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote → {out_dir}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
