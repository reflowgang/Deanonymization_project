from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
REQUIRED_COLUMNS = ['author', 'text', 'timestamp', 'unix_time', 'id', 'parent']
N_CANDIDATE_USERS = 1000
N_QUERY_USERS = 500
N_DISTRACTOR_USERS = 500
RANDOM_SEED = 42

@dataclass(frozen=True)
class Paths:
    raw_csv: Path
    splits_query_full_dir: Path
    splits_candidate_dir: Path
    mapping_csv: Path
    pool_manifest_csv: Path
    truncated_root: Path
    truncation_summary_csv: Path
TRUNCATION_LEVELS: list[tuple[str, Optional[int]]] = [('T1', 5), ('T2', 10), ('T3', 25), ('T4', 50), ('T5', 100), ('T6', 200), ('T7', 500), ('T8', None)]

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_hn_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    print(f'Loading CSV: {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df):,} rows with {len(df.columns)} columns.')
    return df

def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing) + f'. Found columns: {list(df.columns)}')

def normalize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    n_bad = int(df['timestamp'].isna().sum())
    if n_bad:
        print(f'Warning: {n_bad:,} rows have invalid timestamps (set to NaT). Dropping.')
        df = df.dropna(subset=['timestamp']).copy()
    df['author'] = df['author'].astype(str)
    df['text'] = df['text'].astype(str)
    before = len(df)
    df = df[(df['author'].str.len() > 0) & (df['text'].str.len() > 0)].copy()
    dropped = before - len(df)
    if dropped:
        print(f'Dropped {dropped:,} rows with empty author/text.')
    df = df.sort_values(['author', 'timestamp', 'unix_time', 'id'], kind='mergesort')
    df = df.reset_index(drop=True)
    return df

def print_dataset_stats(df: pd.DataFrame) -> None:
    print('\n=== Dataset statistics ===')
    print(f'Rows: {len(df):,}')
    n_authors = int(df['author'].nunique(dropna=False))
    print(f'Unique authors: {n_authors:,}')
    counts = df.groupby('author', dropna=False).size().sort_values(ascending=False)
    if len(counts) == 0:
        print('No authors found after filtering.')
        return
    print(f'Min comments/author: {int(counts.min()):,}')
    print(f'Max comments/author: {int(counts.max()):,}')
    print('\nTop 10 authors by comment count:')
    top10 = counts.head(10)
    for (author, n) in top10.items():
        print(f'- {author}: {int(n):,}')
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    print(f'\nFirst timestamp: {(ts_min.isoformat() if pd.notna(ts_min) else ts_min)}')
    print(f'Last timestamp:  {(ts_max.isoformat() if pd.notna(ts_max) else ts_max)}')
    print('==========================\n')

def make_user_ids(authors: list[str]) -> dict[str, str]:
    width = max(4, int(math.log10(max(1, len(authors)))) + 1)
    return {author: f'user_{i:0{width}d}' for (i, author) in enumerate(authors, start=1)}

def iter_user_rows(df: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for (author, g) in df.groupby('author', sort=False):
        yield (str(author), g)

def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    _ensure_parent_dir(path)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
    return n

def row_to_record(row: pd.Series, user_id: str) -> dict:
    ts = row['timestamp']
    ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
    return {'author_original': row['author'], 'user_id': user_id, 'text': row['text'], 'timestamp': ts_str, 'unix_time': int(row['unix_time']) if pd.notna(row['unix_time']) else None, 'id': row['id'], 'parent': row['parent']}

def split_user_comments(user_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_df = user_df.sort_values(['timestamp', 'unix_time', 'id'], kind='mergesort')
    n = len(user_df)
    cut = n // 2
    query_full = user_df.iloc[:cut].copy()
    candidate = user_df.iloc[cut:].copy()
    return (query_full, candidate)

def tokenize_words(text: str) -> int:
    return len(text.split())

def select_pool_authors(df: pd.DataFrame, min_comments: int) -> tuple[list[str], list[str], list[str]]:
    authors_sorted = df.groupby('author').size().sort_values(ascending=False).index.astype(str).tolist()
    eligible = [author for author in authors_sorted if int((df['author'] == author).sum()) >= min_comments]
    if len(eligible) < N_CANDIDATE_USERS:
        raise ValueError(f'Need at least {N_CANDIDATE_USERS} authors with >= {min_comments} comments; found {len(eligible):,}.')
    candidate_authors = eligible[:N_CANDIDATE_USERS]
    rng = random.Random(RANDOM_SEED)
    query_authors = sorted(rng.sample(candidate_authors, N_QUERY_USERS), key=candidate_authors.index)
    query_author_set = set(query_authors)
    distractor_authors = [a for a in candidate_authors if a not in query_author_set]
    if len(query_authors) != N_QUERY_USERS:
        raise ValueError(f'Expected {N_QUERY_USERS} query authors, got {len(query_authors):,}.')
    if len(distractor_authors) != N_DISTRACTOR_USERS:
        raise ValueError(f'Expected {N_DISTRACTOR_USERS} distractor authors, got {len(distractor_authors):,}.')
    return (candidate_authors, query_authors, distractor_authors)

def validate_pool_author_partition(candidate_authors: list[str], query_authors: list[str], distractor_authors: list[str]) -> None:
    query_set = set(query_authors)
    distractor_set = set(distractor_authors)
    candidate_set = set(candidate_authors)
    if query_set & distractor_set:
        raise ValueError('query_authors and distractor_authors must be disjoint.')
    if query_set | distractor_set != candidate_set:
        raise ValueError('query_authors and distractor_authors must partition candidate_authors.')

def validate_pool_outputs(manifest_df: pd.DataFrame, query_user_ids: set[str], paths: Paths) -> None:
    if len(manifest_df) != N_CANDIDATE_USERS:
        raise ValueError(f'Manifest must have {N_CANDIDATE_USERS} rows, got {len(manifest_df):,}.')
    if int(manifest_df['has_candidate'].sum()) != N_CANDIDATE_USERS:
        raise ValueError('Expected has_candidate == 1 for all 1000 pool users.')
    if int(manifest_df['has_query'].sum()) != N_QUERY_USERS:
        raise ValueError(f"Expected has_query == 1 for {N_QUERY_USERS} users, got {int(manifest_df['has_query'].sum()):,}.")
    if int(manifest_df['is_distractor'].sum()) != N_DISTRACTOR_USERS:
        raise ValueError(f"Expected is_distractor == 1 for {N_DISTRACTOR_USERS} users, got {int(manifest_df['is_distractor'].sum()):,}.")
    query_rows = manifest_df[manifest_df['has_query'] == 1]
    if not (query_rows['has_candidate'] == 1).all():
        raise ValueError('Every query user must also have has_candidate == 1.')
    distractor_rows = manifest_df[manifest_df['is_distractor'] == 1]
    if not (distractor_rows['has_query'] == 0).all():
        raise ValueError('Every distractor must have has_query == 0.')
    for user_id in query_user_ids:
        query_path = paths.splits_query_full_dir / f'{user_id}_query_full.jsonl'
        cand_path = paths.splits_candidate_dir / f'{user_id}_candidate.jsonl'
        if not query_path.exists():
            raise ValueError(f'Missing query_full split for query user: {query_path}')
        if not cand_path.exists():
            raise ValueError(f'Missing candidate split for query user: {cand_path}')
    distractor_ids = set(manifest_df.loc[manifest_df['is_distractor'] == 1, 'user_id'].astype(str))
    for user_id in distractor_ids:
        query_path = paths.splits_query_full_dir / f'{user_id}_query_full.jsonl'
        if query_path.exists():
            raise ValueError(f'Distractor must not have query_full split: {query_path}')
        cand_path = paths.splits_candidate_dir / f'{user_id}_candidate.jsonl'
        if not cand_path.exists():
            raise ValueError(f'Missing candidate split for distractor: {cand_path}')
    for (level, _requested) in TRUNCATION_LEVELS:
        out_dir = paths.truncated_root / level
        n_files = len(list(out_dir.glob('*_query_*.jsonl')))
        if n_files != N_QUERY_USERS:
            raise ValueError(f'Truncation level {level} must contain exactly {N_QUERY_USERS} query files; found {n_files:,} in {out_dir}.')

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(raw_csv=project_root / 'data/raw/hn_comments_active_users_1500.csv', splits_query_full_dir=project_root / 'data/splits/query_full', splits_candidate_dir=project_root / 'data/splits/candidate', mapping_csv=project_root / 'data/filtered/hn_user_mapping.csv', pool_manifest_csv=project_root / 'data/filtered/hn_pool_manifest.csv', truncated_root=project_root / 'data/truncated', truncation_summary_csv=project_root / 'results/tables/hn_truncation_counts.csv')
    df = load_hn_csv(paths.raw_csv)
    validate_columns(df, REQUIRED_COLUMNS)
    df = normalize_and_sort(df)
    print_dataset_stats(df)
    _ensure_dir(paths.splits_query_full_dir)
    _ensure_dir(paths.splits_candidate_dir)
    _ensure_parent_dir(paths.mapping_csv)
    _ensure_parent_dir(paths.pool_manifest_csv)
    _ensure_parent_dir(paths.truncation_summary_csv)
    for (level, _n) in TRUNCATION_LEVELS:
        _ensure_dir(paths.truncated_root / level)
    min_comments = 20
    (candidate_authors, query_authors, distractor_authors) = select_pool_authors(df, min_comments=min_comments)
    query_author_set = set(query_authors)
    distractor_author_set = set(distractor_authors)
    author_to_user_id = make_user_ids(candidate_authors)
    author_groups = {author: g for (author, g) in df.groupby('author', sort=False)}
    print(f'POOL setup (min_comments={min_comments}): candidate pool = top {N_CANDIDATE_USERS} eligible users by activity; query users = random {N_QUERY_USERS} from pool (seed {RANDOM_SEED}); distractors = remaining {N_DISTRACTOR_USERS} candidate-only users.')
    mapping_rows: list[dict] = []
    manifest_rows: list[dict] = []
    query_by_user: dict[str, pd.DataFrame] = {}
    query_user_ids: set[str] = set()
    for (i, author) in enumerate(candidate_authors, start=1):
        g = author_groups[author]
        total = int(len(g))
        user_id = author_to_user_id[author]
        is_query = author in query_author_set
        is_distractor = author in distractor_author_set
        (q_df, c_df) = split_user_comments(g)
        cand_path = paths.splits_candidate_dir / f'{user_id}_candidate.jsonl'
        c_n = write_jsonl(cand_path, (row_to_record(row, user_id) for (_, row) in c_df.iterrows()))
        q_n = 0
        if is_query:
            query_path = paths.splits_query_full_dir / f'{user_id}_query_full.jsonl'
            q_n = write_jsonl(query_path, (row_to_record(row, user_id) for (_, row) in q_df.iterrows()))
            query_by_user[user_id] = q_df.reset_index(drop=True)
            query_user_ids.add(user_id)
        elif is_distractor:
            pass
        else:
            raise ValueError(f'Author {author!r} is neither query nor distractor.')
        mapping_rows.append({'user_id': user_id, 'author_original': author, 'total_comments': total, 'query_comments': q_n, 'candidate_comments': c_n})
        manifest_rows.append({'user_id': user_id, 'author': author, 'role': 'query' if is_query else 'distractor', 'has_query': int(is_query), 'has_candidate': 1, 'is_distractor': int(is_distractor)})
        if i % 100 == 0:
            print(f'- Processed {i:,}/{N_CANDIDATE_USERS:,} pool users...')
    mapping_df = pd.DataFrame(mapping_rows).sort_values('user_id').reset_index(drop=True)
    mapping_df.to_csv(paths.mapping_csv, index=False)
    print(f'\nWrote mapping file: {paths.mapping_csv} ({len(mapping_df):,} users)')
    manifest_df = pd.DataFrame(manifest_rows).sort_values('user_id').reset_index(drop=True)
    manifest_df.to_csv(paths.pool_manifest_csv, index=False)
    print(f'Wrote pool manifest: {paths.pool_manifest_csv} ({len(manifest_df):,} users)')
    print('\nCreating truncation levels from query_full.')
    trunc_summary_rows: list[dict] = []
    for (level, requested) in TRUNCATION_LEVELS:
        out_dir = paths.truncated_root / level
        saved_users = 0
        total_comments_saved = 0
        total_words_saved = 0
        for (user_id, q_df) in query_by_user.items():
            n_q = len(q_df)
            if requested is None:
                t_df = q_df
            else:
                t_df = q_df.iloc[:min(int(requested), n_q)]
            out_path = out_dir / f'{user_id}_query_{level}.jsonl'
            n_written = write_jsonl(out_path, (row_to_record(row, user_id) for (_, row) in t_df.iterrows()))
            saved_users += 1
            total_comments_saved += n_written
            total_words_saved += int(t_df['text'].astype(str).map(tokenize_words).sum())
        avg_comments = total_comments_saved / saved_users if saved_users else 0.0
        avg_words = total_words_saved / total_comments_saved if total_comments_saved else 0.0
        trunc_summary_rows.append({'level': level, 'requested_comments': 'full' if requested is None else int(requested), 'n_users_available': int(saved_users), 'avg_comments': float(avg_comments), 'avg_words': float(avg_words)})
        print(f"- {level}: requested={('full' if requested is None else requested)}, users={saved_users:,}, avg_comments={avg_comments:.2f}, avg_words={avg_words:.2f}")
    summary_df = pd.DataFrame(trunc_summary_rows)
    summary_df.to_csv(paths.truncation_summary_csv, index=False)
    print(f'\nWrote truncation summary: {paths.truncation_summary_csv}')
    print('\nRunning validation checks...')
    validate_pool_author_partition(candidate_authors, query_authors, distractor_authors)
    validate_pool_outputs(manifest_df, query_user_ids, paths)
    print('All validation checks passed.')
    print('\nDone.')
    print(f'- Candidate pool: top {N_CANDIDATE_USERS:,} eligible users by activity; query users: random {N_QUERY_USERS:,} (seed {RANDOM_SEED}); distractors: {N_DISTRACTOR_USERS:,} candidate-only')
    print(f'- Query/candidate JSONL written under: {paths.splits_query_full_dir} and {paths.splits_candidate_dir}')
    print(f'- Truncated query JSONL written under: {paths.truncated_root}/T*/')
if __name__ == '__main__':
    main()
