from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONTENT_TYPES, MIN_COMMENTS_PER_TYPE, PROFILE_COMMENT_COUNT, TYPE_LABELS, paths
from io_utils import resolve_run_paths

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build P/O/T profile text files from classifications.')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--min-comments', type=int, default=MIN_COMMENTS_PER_TYPE, help='Minimum comments of a type required to write a profile file.')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    run_paths = resolve_run_paths(pilot=args.pilot)
    classifications_csv = run_paths['classifications_csv']
    profiles_root = run_paths['type_profiles_root']
    min_comments = int(args.min_comments)
    if not classifications_csv.exists():
        raise FileNotFoundError(f'Classifications not found: {classifications_csv}')
    df = pd.read_csv(classifications_csv)
    required = {'user_id', 'comment_index', 'timestamp', 'label', 'body'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Classifications CSV missing columns: {sorted(missing)}')
    df['user_id'] = df['user_id'].astype(str)
    df['label'] = df['label'].astype(str).str.upper()
    df = df[df['label'].isin(CONTENT_TYPES)].copy()
    df = df.sort_values(['user_id', 'timestamp', 'comment_index'], kind='mergesort')
    for label in CONTENT_TYPES:
        (profiles_root / TYPE_LABELS[label]).mkdir(parents=True, exist_ok=True)
    built_counts = {label: 0 for label in CONTENT_TYPES}
    qualified_users: set[str] = set()
    for (user_id, user_df) in df.groupby('user_id', sort=True):
        counts = user_df['label'].value_counts().reindex(CONTENT_TYPES, fill_value=0).astype(int)
        qualifies_all = counts['P'] >= min_comments and counts['O'] >= min_comments and (counts['T'] >= min_comments)
        if not qualifies_all:
            continue
        qualified_users.add(str(user_id))
        for label in CONTENT_TYPES:
            typed = user_df[user_df['label'] == label]
            selected = typed.head(PROFILE_COMMENT_COUNT)
            out_path = profiles_root / TYPE_LABELS[label] / f'{user_id}.txt'
            with out_path.open('w', encoding='utf-8') as f:
                for body in selected['body']:
                    line = str(body).replace('\n', ' ').strip()
                    if line:
                        f.write(line + '\n')
            built_counts[label] += 1
    for label in CONTENT_TYPES:
        type_dir = profiles_root / TYPE_LABELS[label]
        if not type_dir.exists():
            continue
        for stale in type_dir.glob('*.txt'):
            if stale.stem not in qualified_users:
                stale.unlink()
    print(f'Built type-filtered profiles under: {profiles_root}')
    print(f'Minimum comments per type required: {min_comments}')
    for label in CONTENT_TYPES:
        n_files = len(list((profiles_root / TYPE_LABELS[label]).glob('*.txt')))
        print(f'- {label} ({TYPE_LABELS[label]}): {built_counts[label]:,} profiles written ({n_files:,} on disk)')
if __name__ == '__main__':
    main()
