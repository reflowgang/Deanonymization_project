from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esrc.config import load_dotenv_files
from esrc.paths import bsp_candidate_embeddings_index_csv, bsp_faiss_top15, bsp_hn_candidate_summaries, bsp_hn_faiss_top15, bsp_pool_en_candidate_summaries, list_hn_query_user_ids, list_pool_en_query_user_ids, load_hn_candidate_summary_text, summer_root
from esrc.search import search_top_k
from jina_clients import embed_texts_jina, rerank_documents
DEFAULT_P1_RUN = summer_root() / 'results' / 'runs' / 'p1_full_pool_overnight_20260805'
DEFAULT_OUT = summer_root() / 'results' / 'x_embedder_check'
K = 15

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Jina embed/rerank search side check')
    p.add_argument('--p1-run-dir', type=str, default=str(DEFAULT_P1_RUN))
    p.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT))
    p.add_argument('--pool', choices=('pool_en', 'hn', 'both'), default='both')
    p.add_argument('--skip-rerank', action='store_true')
    p.add_argument('--embed-batch-size', type=int, default=64)
    return p.parse_args()

def load_pool_en_candidate_texts() -> tuple[list[str], list[str]]:
    idx_path = bsp_candidate_embeddings_index_csv()
    user_ids: list[str] = []
    with idx_path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            user_ids.append(row['user_id'])
    cand_dir = bsp_pool_en_candidate_summaries()
    texts: list[str] = []
    for uid in user_ids:
        path = cand_dir / f'{uid}.txt'
        if not path.exists():
            raise FileNotFoundError(f'Missing pool_en candidate summary: {path}')
        texts.append(path.read_text(encoding='utf-8').strip())
    return (user_ids, texts)

def load_hn_candidate_texts() -> tuple[list[str], list[str]]:
    paths = sorted(bsp_hn_candidate_summaries().glob('user_*_summary.json'))
    if not paths:
        raise FileNotFoundError('No HN candidate summaries found')
    user_ids: list[str] = []
    texts: list[str] = []
    for path in paths:
        stem = path.stem
        uid = stem[:-len('_summary')] if stem.endswith('_summary') else stem
        user_ids.append(uid)
        texts.append(load_hn_candidate_summary_text(uid))
    return (user_ids, texts)

def load_or_build_candidate_index(pool: str, out_pool_dir: Path, *, batch_size: int) -> tuple[list[str], np.ndarray]:
    npy_path = out_pool_dir / 'candidate_embeddings_jina.npy'
    idx_path = out_pool_dir / 'candidate_embeddings_index.csv'
    if npy_path.exists() and idx_path.exists():
        user_ids: list[str] = []
        with idx_path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                user_ids.append(row['user_id'])
        matrix = np.load(npy_path)
        if matrix.shape[0] == len(user_ids):
            print(f'[{pool}] candidate jina cache hit ({len(user_ids)} rows, dim={matrix.shape[1]})')
            return (user_ids, matrix)
        print(f'[{pool}] candidate cache shape mismatch; rebuilding')
    if pool == 'pool_en':
        (user_ids, texts) = load_pool_en_candidate_texts()
    else:
        (user_ids, texts) = load_hn_candidate_texts()
    print(f'[{pool}] embedding {len(texts)} candidate summaries with jina…')
    t0 = time.perf_counter()
    matrix = embed_texts_jina(texts, batch_size=batch_size)
    print(f'[{pool}] candidate embed done {matrix.shape} in {time.perf_counter() - t0:.1f}s')
    out_pool_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, matrix)
    with idx_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['row_id', 'user_id'])
        w.writeheader()
        for (i, uid) in enumerate(user_ids):
            w.writerow({'row_id': i, 'user_id': uid})
    return (user_ids, matrix)

def load_query_summaries(extract_dir: Path, users: list[str]) -> tuple[list[str], list[str]]:
    ok_users: list[str] = []
    summaries: list[str] = []
    for uid in users:
        path = extract_dir / f'{uid}.txt'
        if not path.exists():
            print(f'WARN skip {uid}: no extract summary')
            continue
        ok_users.append(uid)
        summaries.append(path.read_text(encoding='utf-8').strip())
    return (ok_users, summaries)

def write_search_csv(path: Path, users: list[str], hits: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['query_user_id', 'rank', 'candidate_user_id', 'score'])
        w.writeheader()
        for (uid, user_hits) in zip(users, hits):
            for h in user_hits:
                w.writerow({'query_user_id': uid, 'rank': h.rank, 'candidate_user_id': h.candidate_user_id, 'score': f'{h.score:.6f}'})
    tmp.replace(path)

def load_search_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by[row['query_user_id']].append(row)
    for uid in by:
        by[uid].sort(key=lambda r: int(r['rank']))
    return by

def metrics_from_search(by_query: dict[str, list[dict[str, str]]]) -> dict[str, float | int]:
    n = len(by_query)
    if n == 0:
        return {'n': 0, 'hit_at_15': float('nan'), 'top1': float('nan')}
    hit15 = 0
    top1 = 0
    for (uid, rows) in by_query.items():
        cands = [r['candidate_user_id'] for r in rows[:K]]
        if uid in cands:
            hit15 += 1
        if cands and cands[0] == uid:
            top1 += 1
    return {'n': n, 'hit_at_15': round(hit15 / n, 4), 'hit_at_15_count': hit15, 'top1': round(top1 / n, 4), 'top1_count': top1}

def load_archived_gpt4o_hits(pool: str) -> dict[str, bool]:
    path = bsp_faiss_top15() if pool == 'pool_en' else bsp_hn_faiss_top15()
    by = load_search_rows(path)
    return {uid: uid in [r['candidate_user_id'] for r in rows[:K]] for (uid, rows) in by.items()}

def rerank_search(*, pool: str, users: list[str], query_summaries: list[str], search_rows: dict[str, list[dict[str, str]]], cand_text_by_id: dict[str, str], out_path: Path) -> dict[str, list[dict[str, str]]]:
    from esrc.search import SearchHit
    out_rows: dict[str, list[dict[str, str]]] = {}
    all_hits: list[list[SearchHit]] = []
    t0 = time.perf_counter()
    for (i, uid) in enumerate(users):
        rows = search_rows[uid][:K]
        doc_ids = [r['candidate_user_id'] for r in rows]
        documents = [cand_text_by_id[cid] for cid in doc_ids]
        ranked = rerank_documents(query_summaries[i], documents)
        reranked_hits: list[SearchHit] = []
        for (rank, (orig_idx, score)) in enumerate(ranked, start=1):
            reranked_hits.append(SearchHit(candidate_user_id=doc_ids[orig_idx], rank=rank, score=score))
        all_hits.append(reranked_hits)
        out_rows[uid] = [{'query_user_id': uid, 'rank': str(h.rank), 'candidate_user_id': h.candidate_user_id, 'score': f'{h.score:.6f}'} for h in reranked_hits]
        if (i + 1) % 50 == 0 or i + 1 == len(users):
            print(f'[{pool}] rerank {i + 1}/{len(users)} …', flush=True)
    write_search_csv(out_path, users, all_hits)
    print(f'[{pool}] rerank done in {time.perf_counter() - t0:.1f}s → {out_path}')
    return out_rows

def run_pool(pool: str, *, p1_dir: Path, out_dir: Path, batch_size: int, skip_rerank: bool) -> dict:
    out_pool = out_dir / pool
    out_pool.mkdir(parents=True, exist_ok=True)
    users = list_pool_en_query_user_ids() if pool == 'pool_en' else list_hn_query_user_ids()
    extract_dir = p1_dir / pool / 'extract'
    (ok_users, summaries) = load_query_summaries(extract_dir, users)
    (cand_ids, cand_matrix) = load_or_build_candidate_index(pool, out_pool, batch_size=batch_size)
    print(f'[{pool}] embedding {len(ok_users)} query summaries with jina…')
    t0 = time.perf_counter()
    q_vecs = embed_texts_jina(summaries, batch_size=batch_size)
    print(f'[{pool}] query embed done in {time.perf_counter() - t0:.1f}s')
    hits = search_top_k(q_vecs, cand_matrix, cand_ids, k=K)
    jina_path = out_pool / 'search_top15_jina.csv'
    write_search_csv(jina_path, ok_users, hits)
    jina_by = load_search_rows(jina_path)
    mpnet_path = p1_dir / pool / 'search_top15.csv'
    mpnet_by = load_search_rows(mpnet_path)
    mpnet_by = {u: mpnet_by[u] for u in ok_users if u in mpnet_by}
    cand_text_by_id: dict[str, str] = {}
    if pool == 'pool_en':
        cand_dir = bsp_pool_en_candidate_summaries()
        for cid in cand_ids:
            cand_text_by_id[cid] = (cand_dir / f'{cid}.txt').read_text(encoding='utf-8').strip()
    else:
        for cid in cand_ids:
            cand_text_by_id[cid] = load_hn_candidate_summary_text(cid)
    rerank_by: dict[str, list[dict[str, str]]] | None = None
    if not skip_rerank:
        rerank_path = out_pool / 'search_top15_jina_rerank.csv'
        rerank_by = rerank_search(pool=pool, users=ok_users, query_summaries=summaries, search_rows=jina_by, cand_text_by_id=cand_text_by_id, out_path=rerank_path)
    gpt4o_all = load_archived_gpt4o_hits(pool)
    gpt4o_hits = sum((1 for u in ok_users if gpt4o_all.get(u, False)))
    gpt4o_n = sum((1 for u in ok_users if u in gpt4o_all))
    result = {'pool': pool, 'n_queries': len(ok_users), 'mpnet_baseline': metrics_from_search(mpnet_by), 'jina_embed': metrics_from_search(jina_by), 'gpt4o_archived': {'n': gpt4o_n, 'hit_at_15': round(gpt4o_hits / gpt4o_n, 4) if gpt4o_n else None, 'hit_at_15_count': gpt4o_hits}}
    if rerank_by is not None:
        result['jina_embed_rerank'] = metrics_from_search(rerank_by)
    metrics_path = out_pool / 'metrics.json'
    metrics_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"[{pool}] mpnet Hit@15={result['mpnet_baseline']['hit_at_15']:.1%} jina={result['jina_embed']['hit_at_15']:.1%}" + (f" rerank={result['jina_embed_rerank']['hit_at_15']:.1%}" if rerank_by is not None else '') + f" gpt4o={result['gpt4o_archived']['hit_at_15']:.1%}")
    return result

def write_summary(out_dir: Path, rows: list[dict]) -> None:
    lines = ['# x_embedder_check — jina embed / rerank side check', '', f'Generated {datetime.now(timezone.utc).isoformat()}', '', 'Reads P1 baseline local Extract summaries only; all new outputs under `results/x_embedder_check/`.', '', '| Pool | Method | n | Hit@15 | top-1 |', '|------|--------|---|--------|-------|']
    for r in rows:
        pool = r['pool']
        for (key, label) in (('mpnet_baseline', 'mpnet (P1 baseline)'), ('jina_embed', 'jina-embeddings-v3'), ('jina_embed_rerank', 'jina embed + rerank'), ('gpt4o_archived', 'gpt-4o archived (BSP)')):
            block = r.get(key)
            if not block:
                continue
            n = block.get('n')
            hit = block.get('hit_at_15')
            top1 = block.get('top1')
            if hit is None and 'hit_at_15' in block:
                hit = block['hit_at_15']
            if top1 is None:
                top1 = float('nan')
            hit_s = f'{100 * hit:.1f}%' if isinstance(hit, (int, float)) and hit == hit else '—'
            top1_s = f'{100 * top1:.1f}%' if isinstance(top1, (int, float)) and top1 == top1 else '—'
            lines.append(f'| {pool} | {label} | {n} | {hit_s} | {top1_s} |')
    (out_dir / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> int:
    args = parse_args()
    load_dotenv_files(summer_root())
    load_dotenv_files(summer_root().parent)
    p1_dir = Path(args.p1_run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = ['pool_en', 'hn'] if args.pool == 'both' else [args.pool]
    results: list[dict] = []
    t_all = time.perf_counter()
    for pool in pools:
        results.append(run_pool(pool, p1_dir=p1_dir, out_dir=out_dir, batch_size=args.embed_batch_size, skip_rerank=args.skip_rerank))
    write_summary(out_dir, results)
    manifest = {'created': datetime.now(timezone.utc).isoformat(), 'p1_run_dir': str(p1_dir), 'out_dir': str(out_dir), 'pools': results, 'elapsed_s': round(time.perf_counter() - t_all, 1)}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f"Done in {manifest['elapsed_s']:.1f}s → {out_dir}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
