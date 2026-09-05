from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.confidence import estimator_b_selected_id_logprob
from esrc.config import load_dotenv_files
from esrc.generate import generate, get_client
from esrc.manifests import git_commit, sha256_file, write_manifest
from esrc.paths import bsp_prompt_record_selection, summer_root
from esrc.reason_prompt import build_candidate_block, build_user_prompt, clamp01, extract_json_object, load_prompt_template, resolve_predicted_candidate_id
DEFAULT_REASON_MODEL = 'qwen3.6-35b-a3b-nvfp4'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='P2.3 Reason pick on fixture inputs')
    p.add_argument('--inputs-dir', type=str, default=str(summer_root() / 'results/runs/p2_inputs_regression_50_T8/inputs'))
    p.add_argument('--model', type=str, default=os.getenv('VLLM_REASON_MODEL', DEFAULT_REASON_MODEL))
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--out-dir', type=str, default=None, help='Default: results/runs/p2_reason_<timestamp>/')
    p.add_argument('--resume', action='store_true', help='Skip query_user_ids already ok in existing reason_predictions.jsonl')
    return p.parse_args()

def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('status') == 'ok':
                done.add(obj['query_user_id'])
    return done

def main() -> int:
    load_dotenv_files(summer_root())
    args = parse_args()
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.is_dir():
        raise SystemExit(f'Inputs dir not found: {inputs_dir}. Run 01_build_fixture_reason_inputs.py first.')
    packs = sorted(inputs_dir.glob('user_*.json'))
    if args.limit is not None:
        packs = packs[:args.limit]
    if not packs:
        raise SystemExit(f'No packs in {inputs_dir}')
    out_dir = Path(args.out_dir) if args.out_dir else summer_root() / 'results' / 'runs' / f"p2_reason_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / 'reason_predictions.jsonl'
    done = load_done(out_jsonl) if args.resume else set()
    prompt_path = bsp_prompt_record_selection()
    template = load_prompt_template(prompt_path)
    client = get_client()
    print(f'model={args.model} packs={len(packs)} out={out_dir}')
    print(f'enable_thinking=False (locked for Reason / P2)')
    n_ok = n_err = 0
    for pack_path in packs:
        pack = json.loads(pack_path.read_text(encoding='utf-8'))
        uid = pack['query_user_id']
        if uid in done:
            continue
        cand_ids = [c['candidate_user_id'] for c in pack['candidates']]
        block = build_candidate_block(pack['candidates'])
        user_prompt = build_user_prompt(template, pack['query_summary'], block)
        row: dict = {'fixture_id': pack.get('fixture_id'), 'T_level': pack.get('T_level'), 'query_user_id': uid, 'candidate_user_ids': cand_ids, 'model': args.model, 'enable_thinking': False, 'status': 'error', 'error': ''}
        try:
            result = generate([{'role': 'user', 'content': user_prompt}], model=args.model, client=client, temperature=0.0, max_tokens=args.max_tokens, logprobs=True, top_logprobs=5, seed=args.seed, enable_thinking=False)
            if not result.text and result.reasoning_content:
                raise ValueError('Empty content with non-empty reasoning_content — thinking not disabled?')
            obj = extract_json_object(result.text)
            (pred_id, pred_num, err) = resolve_predicted_candidate_id(obj, cand_ids)
            if err:
                raise ValueError(err)
            conf = clamp01(obj.get('confidence', 0.0))
            id_lp = estimator_b_selected_id_logprob(result.token_logprobs, pred_num)
            row.update({'selected_candidate_user_id': pred_id, 'selected_candidate_number': pred_num, 'verbalized_confidence': conf, 'reasoning_short': str(obj.get('reasoning_short', ''))[:500], 'correct': pred_id == uid, 'sequence_logprob_full': result.sequence_logprob, 'selected_id_logprob': id_lp, 'n_token_logprobs': len(result.token_logprobs), 'raw_text': result.text, 'reasoning_content_len': len(result.reasoning_content or ''), 'finish_reason': result.finish_reason, 'status': 'ok', 'error': ''})
            row['token_logprobs'] = [{'token': t.token, 'logprob': t.logprob} for t in result.token_logprobs]
            n_ok += 1
            print(f"ok {uid} pick={pred_num} conf={conf:.3f} correct={row['correct']}")
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            n_err += 1
            print(f"ERR {uid}: {row['error']}")
        with out_jsonl.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_manifest(out_dir / 'manifest.json', {'run_id': out_dir.name, 'task': 'p2.3_reason_pick', 'model': args.model, 'seed': args.seed, 'enable_thinking': False, 'prompt_file': str(prompt_path), 'prompt_sha256': sha256_file(prompt_path), 'inputs_dir': str(inputs_dir), 'n_ok': n_ok, 'n_error': n_err, 'git_commit': git_commit(summer_root())})
    print(f'Done. ok={n_ok} err={n_err} → {out_jsonl}')
    return 0 if n_err == 0 else 1
if __name__ == '__main__':
    raise SystemExit(main())
