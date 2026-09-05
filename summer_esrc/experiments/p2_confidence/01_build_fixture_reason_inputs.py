from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from esrc.config import load_dotenv_files
from esrc.manifests import git_commit, sha256_file, sha256_text, write_manifest
from esrc.paths import bsp_faiss_top15, bsp_pool_en, bsp_prompt_record_selection, fixture_user_ids, summer_root

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build P2 fixture Reason inputs from BSP')
    p.add_argument('--fixture-id', default='regression_50')
    p.add_argument('--level', default='T8')
    p.add_argument('--out-dir', type=str, default=None, help='Default: results/runs/p2_inputs_<fixture>_<level>/')
    return p.parse_args()

def main() -> int:
    load_dotenv_files(summer_root())
    args = parse_args()
    users = fixture_user_ids(args.fixture_id)
    user_set = set(users)
    level = args.level
    faiss_path = bsp_faiss_top15()
    summ_root = bsp_pool_en() / 'summaries' / level
    cand_summ_dir = bsp_pool_en() / 'candidate_summaries'
    prompt_path = bsp_prompt_record_selection()
    by_query: dict[str, list[dict]] = {u: [] for u in users}
    with faiss_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['T_level'] != level:
                continue
            qid = row['query_user_id']
            if qid not in user_set:
                continue
            by_query[qid].append({'candidate_user_id': row['candidate_user_id'], 'rank': int(row['rank']), 'score': float(row['score'])})
    missing = [u for u in users if len(by_query[u]) == 0]
    if missing:
        raise SystemExit(f'No FAISS rows for {len(missing)} users, e.g. {missing[:3]}')
    out_dir = Path(args.out_dir) if args.out_dir else summer_root() / 'results' / 'runs' / f'p2_inputs_{args.fixture_id}_{level}'
    inputs_dir = out_dir / 'inputs'
    inputs_dir.mkdir(parents=True, exist_ok=True)
    packs = []
    for uid in users:
        cands = sorted(by_query[uid], key=lambda r: r['rank'])
        if len(cands) != 15:
            print(f'WARN {uid}: expected 15 candidates, got {len(cands)}')
        q_path = summ_root / f'{uid}.txt'
        if not q_path.exists():
            raise SystemExit(f'Missing query summary: {q_path}')
        q_text = q_path.read_text(encoding='utf-8').strip()
        enriched = []
        for c in cands:
            cid = c['candidate_user_id']
            c_path = cand_summ_dir / f'{cid}.txt'
            if not c_path.exists():
                raise SystemExit(f'Missing candidate summary: {c_path}')
            enriched.append({**c, 'summary': c_path.read_text(encoding='utf-8').strip(), 'summary_sha256': sha256_file(c_path)})
        pack = {'fixture_id': args.fixture_id, 'T_level': level, 'query_user_id': uid, 'query_summary': q_text, 'query_summary_sha256': sha256_file(q_path), 'candidates': enriched, 'true_in_top15': any((c['candidate_user_id'] == uid for c in enriched))}
        pack_path = inputs_dir / f'{uid}.json'
        pack_path.write_text(json.dumps(pack, indent=2) + '\n', encoding='utf-8')
        packs.append(uid)
    man = {'run_id': out_dir.name, 'task': 'p2.2_build_fixture_reason_inputs', 'fixture_id': args.fixture_id, 'T_level': level, 'n_users': len(packs), 'source': 'bsp_faiss_pool_en', 'note': 'No P1.4/P1.5 local Search artifacts under summer_esrc/results/runs/; using BSP FAISS top-15 + summaries (read-only).', 'faiss_csv': str(faiss_path), 'faiss_sha256': sha256_file(faiss_path), 'prompt_file': str(prompt_path), 'prompt_sha256': sha256_file(prompt_path), 'user_ids_sha256': sha256_text((summer_root() / 'data/fixtures' / args.fixture_id / 'user_ids.txt').read_text(encoding='utf-8')), 'git_commit': git_commit(summer_root()), 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    write_manifest(out_dir / 'manifest.json', man)
    print(f'Wrote {len(packs)} packs → {inputs_dir}')
    print(f"Manifest → {out_dir / 'manifest.json'}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
