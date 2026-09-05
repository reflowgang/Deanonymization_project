from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONTENT_TYPES, MIN_COMMENTS_PER_TYPE, PROFILE_COMMENT_COUNT, TYPE_LABELS
from io_utils import resolve_run_paths

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Report P/O/T qualification statistics.')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--extrapolate-users', type=int, default=500, help='Total user population for full-run qualification extrapolation.')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    run_paths = resolve_run_paths(pilot=args.pilot)
    classifications_csv = run_paths['classifications_csv']
    qualification_csv = run_paths['qualification_csv']
    summary_csv = run_paths['summary_csv']
    profiles_root = run_paths['type_profiles_root']
    if not classifications_csv.exists():
        raise FileNotFoundError(f'Classifications not found: {classifications_csv}')
    df = pd.read_csv(classifications_csv)
    df['user_id'] = df['user_id'].astype(str)
    df['label'] = df['label'].astype(str).str.upper()
    users_processed = int(df['user_id'].nunique())
    total_comments = int(len(df))
    label_counts = df['label'].value_counts().reindex(CONTENT_TYPES, fill_value=0).astype(int)
    qual_rows: list[dict[str, object]] = []
    for (user_id, user_df) in df.groupby('user_id', sort=True):
        counts = user_df['label'].value_counts().reindex(CONTENT_TYPES, fill_value=0).astype(int)
        qual_rows.append({'user_id': user_id, 'n_comments': int(len(user_df)), 'n_personal': int(counts['P']), 'n_opinion': int(counts['O']), 'n_topical': int(counts['T']), 'qualifies_personal': int(counts['P'] >= MIN_COMMENTS_PER_TYPE), 'qualifies_opinion': int(counts['O'] >= MIN_COMMENTS_PER_TYPE), 'qualifies_topical': int(counts['T'] >= MIN_COMMENTS_PER_TYPE), 'qualifies_all_three': int(counts['P'] >= MIN_COMMENTS_PER_TYPE and counts['O'] >= MIN_COMMENTS_PER_TYPE and (counts['T'] >= MIN_COMMENTS_PER_TYPE))})
    qual_df = pd.DataFrame(qual_rows)
    qualification_csv.parent.mkdir(parents=True, exist_ok=True)
    qual_df.to_csv(qualification_csv, index=False)
    n_qual_p = int(qual_df['qualifies_personal'].sum())
    n_qual_o = int(qual_df['qualifies_opinion'].sum())
    n_qual_t = int(qual_df['qualifies_topical'].sum())
    n_qual_all = int(qual_df['qualifies_all_three'].sum())
    profile_counts = {label: len(list((profiles_root / TYPE_LABELS[label]).glob('*.txt'))) for label in CONTENT_TYPES}
    log_csv = run_paths['classify_log_csv']
    api_stats = {'batch_ok': 0, 'batch_error': 0, 'single_fallback_ok': 0, 'single_fallback_error': 0}
    if log_csv.exists():
        log_df = pd.read_csv(log_csv)
        api_stats['batch_ok'] = int(((log_df['mode'] == 'batch') & (log_df['status'] == 'ok')).sum())
        api_stats['batch_error'] = int(((log_df['mode'] == 'batch') & (log_df['status'] == 'error')).sum())
        api_stats['single_fallback_ok'] = int(((log_df['mode'] == 'single_fallback') & (log_df['status'] == 'ok')).sum())
        api_stats['single_fallback_error'] = int(((log_df['mode'] == 'single_fallback') & (log_df['status'] == 'error')).sum())
    extrapolation_factor = args.extrapolate_users / users_processed if users_processed else 0.0
    est_full_qual_all = int(round(n_qual_all * extrapolation_factor))
    summary_rows = [{'metric': 'users_processed', 'value': users_processed}, {'metric': 'total_comments_classified', 'value': total_comments}, {'metric': 'label_P', 'value': int(label_counts['P'])}, {'metric': 'label_O', 'value': int(label_counts['O'])}, {'metric': 'label_T', 'value': int(label_counts['T'])}, {'metric': 'pct_P', 'value': round(100.0 * label_counts['P'] / total_comments, 2)}, {'metric': 'pct_O', 'value': round(100.0 * label_counts['O'] / total_comments, 2)}, {'metric': 'pct_T', 'value': round(100.0 * label_counts['T'] / total_comments, 2)}, {'metric': 'qualifies_personal_ge_50', 'value': n_qual_p}, {'metric': 'qualifies_opinion_ge_50', 'value': n_qual_o}, {'metric': 'qualifies_topical_ge_50', 'value': n_qual_t}, {'metric': 'qualifies_all_three_ge_50', 'value': n_qual_all}, {'metric': 'profile_files_personal', 'value': profile_counts['P']}, {'metric': 'profile_files_opinion', 'value': profile_counts['O']}, {'metric': 'profile_files_topical', 'value': profile_counts['T']}, {'metric': 'api_batch_ok', 'value': api_stats['batch_ok']}, {'metric': 'api_batch_error', 'value': api_stats['batch_error']}, {'metric': 'api_single_fallback_ok', 'value': api_stats['single_fallback_ok']}, {'metric': 'api_single_fallback_error', 'value': api_stats['single_fallback_error']}, {'metric': 'estimated_full_run_users_qualifying_all_three', 'value': est_full_qual_all}, {'metric': 'estimated_full_run_qualification_rate_pct', 'value': round(100.0 * n_qual_all / users_processed, 2) if users_processed else 0.0}]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print('Content-type qualification report')
    print(f'- users processed: {users_processed:,}')
    print(f'- total comments classified: {total_comments:,}')
    print(f"- label counts: P={int(label_counts['P']):,}, O={int(label_counts['O']):,}, T={int(label_counts['T']):,}")
    print(f"- label shares: P={100 * label_counts['P'] / total_comments:.1f}%, O={100 * label_counts['O'] / total_comments:.1f}%, T={100 * label_counts['T'] / total_comments:.1f}%")
    print(f'- qualifies personal (>=50): {n_qual_p:,}')
    print(f'- qualifies opinion (>=50): {n_qual_o:,}')
    print(f'- qualifies topical (>=50): {n_qual_t:,}')
    print(f'- qualifies all three (>=50 each): {n_qual_all:,} (ESRC cohort)')
    print(f"- profile files built: P={profile_counts['P']}, O={profile_counts['O']}, T={profile_counts['T']}")
    print(f"- API log: batch_ok={api_stats['batch_ok']}, batch_error={api_stats['batch_error']}, single_fallback_ok={api_stats['single_fallback_ok']}, single_fallback_error={api_stats['single_fallback_error']}")
    print(f'- estimated full-run qualification (all three): {est_full_qual_all:,} / {args.extrapolate_users:,} ({100.0 * n_qual_all / users_processed:.1f}% in pilot)')
    print(f'\nWrote: {qualification_csv}')
    print(f'Wrote: {summary_csv}')
if __name__ == '__main__':
    main()
