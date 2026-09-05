from __future__ import annotations
import csv
import html
import json
import re
from pathlib import Path
from esrc.config import project_root

def summer_root() -> Path:
    return project_root()

def bsp_root() -> Path:
    return summer_root().parent

def bsp_prompt_record_selection() -> Path:
    return bsp_root() / 'prompts' / 'record_selection_lermen_g2.txt'

def bsp_prompt_summarization() -> Path:
    return bsp_root() / 'prompts' / 'summarization_lermen_g2.txt'

def summer_prompt_extract_merge() -> Path:
    return summer_root() / 'prompts' / 'extract_merge.txt'

def bsp_truncated_query(level: str, user_id: str) -> Path:
    return bsp_pool_en() / 'truncated_queries' / level / f'{user_id}.txt'

def bsp_candidate_embeddings_npy() -> Path:
    return bsp_pool_en() / 'embeddings' / 'candidate_embeddings.npy'

def bsp_candidate_embeddings_index_csv() -> Path:
    return bsp_root() / 'results' / 'tables' / 'pool_en_candidate_embeddings_index.csv'

def bsp_pool_en() -> Path:
    return bsp_root() / 'data' / 'esrc' / 'pool_en'

def bsp_faiss_top15() -> Path:
    return bsp_root() / 'results' / 'tables' / 'pool_en_faiss_top15.csv'

def bsp_pool_en_candidate_summaries() -> Path:
    return bsp_pool_en() / 'candidate_summaries'

def bsp_hn_truncated_dir(level: str) -> Path:
    return bsp_root() / 'data' / 'truncated' / level

def bsp_hn_user_mapping() -> Path:
    return bsp_root() / 'data' / 'filtered' / 'hn_user_mapping.csv'

def bsp_hn_pool_manifest() -> Path:
    return bsp_root() / 'data' / 'filtered' / 'hn_pool_manifest.csv'

def bsp_hn_candidate_summaries() -> Path:
    return bsp_root() / 'data' / 'extracted_summaries' / 'candidate'

def bsp_hn_faiss_top15() -> Path:
    return bsp_root() / 'results' / 'tables' / 'hn_faiss_top15.csv'

def bsp_hn_candidate_embeddings_index() -> Path:
    return bsp_root() / 'results' / 'tables' / 'hn_candidate_embeddings_index.csv'

def fixture_dir(fixture_id: str='regression_50') -> Path:
    return summer_root() / 'data' / 'fixtures' / fixture_id

def fixture_user_ids(fixture_id: str='regression_50') -> list[str]:
    path = fixture_dir(fixture_id) / 'user_ids.txt'
    lines = path.read_text(encoding='utf-8').splitlines()
    return [ln.strip() for ln in lines if ln.strip()]

def list_pool_en_query_user_ids(level: str='T8') -> list[str]:
    d = bsp_pool_en() / 'truncated_queries' / level
    if not d.exists():
        raise FileNotFoundError(f'Missing pool_en truncated queries: {d}')
    return sorted((p.stem for p in d.glob('user_*.txt')))

def list_hn_query_user_ids(level: str='T8') -> list[str]:
    manifest = bsp_hn_pool_manifest()
    if manifest.exists():
        ids: list[str] = []
        with manifest.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('role') == 'query' or row.get('has_query') == '1':
                    ids.append(row['user_id'])
        if ids:
            return sorted(ids)
    d = bsp_hn_truncated_dir(level)
    if not d.exists():
        raise FileNotFoundError(f'Missing HN truncated queries: {d}')
    out: list[str] = []
    for p in sorted(d.glob('user_*_query_*.jsonl')):
        out.append(p.name.split('_query_')[0])
    return out

def _clean_comment_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub('<[^>]+>', ' ', text)
    return ' '.join(text.split())

def load_hn_query_profile_text(user_id: str, level: str='T8') -> str:
    path = bsp_hn_truncated_dir(level) / f'{user_id}_query_{level}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'Missing HN truncated query: {path}')
    comments: list[str] = []
    with path.open(encoding='utf-8') as f:
        for (line_no, line) in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON on line {line_no} in {path}: {e}') from e
            if 'text' not in obj:
                raise ValueError(f'Missing "text" on line {line_no} in {path}')
            cleaned = _clean_comment_text(str(obj['text']))
            if cleaned:
                comments.append(cleaned)
    return '\n'.join(comments)

def load_hn_candidate_summary_text(user_id: str) -> str:
    path = bsp_hn_candidate_summaries() / f'{user_id}_summary.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing HN candidate summary: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(obj, dict):
        summary = obj.get('summary')
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)
