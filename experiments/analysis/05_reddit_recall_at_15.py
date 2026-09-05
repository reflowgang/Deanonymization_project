from __future__ import annotations
from pathlib import Path
import pandas as pd
LEVELS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
COMMENT_COUNTS = [5, 10, 25, 50, 100, 200, 500, 500]
REDDIT_TOP1 = {'T1': 8.8, 'T2': 13.4, 'T3': 16.6, 'T4': 25.2, 'T5': 28.2, 'T6': 33.2, 'T7': 39.8, 'T8': 39.4}
HN_TOP1 = {'T1': 4.8, 'T2': 8.6, 'T3': 15.6, 'T4': 22.2, 'T5': 26.0, 'T6': 32.4, 'T7': 34.6, 'T8': 38.0}

def recall_at_15_reddit(faiss_df: pd.DataFrame, level: str) -> tuple[float, int]:
    sub = faiss_df[faiss_df['T_level'] == level].copy()
    n_queries = int(sub['query_user_id'].nunique())
    if n_queries == 0:
        return (0.0, 0)
    hits = sub.assign(is_hit=sub['candidate_user_id'].astype(str) == sub['query_user_id'].astype(str)).groupby('query_user_id', as_index=False)['is_hit'].max()['is_hit']
    return (float(hits.mean()), n_queries)

def recall_at_15_hn(faiss_df: pd.DataFrame, level: str) -> tuple[float, int]:
    sub = faiss_df[faiss_df['level'] == level].copy()
    n_queries = int(sub['query_user_id'].nunique())
    if n_queries == 0:
        return (0.0, 0)
    hits = sub.groupby('query_user_id', as_index=False)['is_true_match'].max()['is_true_match']
    return (float(hits.mean()), n_queries)

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tables_dir = project_root / 'results/tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    reddit_faiss_path = tables_dir / 'pool_en_faiss_top15.csv'
    hn_faiss_path = tables_dir / 'hn_faiss_top15.csv'
    reddit_faiss = pd.read_csv(reddit_faiss_path)
    hn_faiss = pd.read_csv(hn_faiss_path)
    reddit_rows: list[dict[str, object]] = []
    for (level, count) in zip(LEVELS, COMMENT_COUNTS):
        (recall, n_queries) = recall_at_15_reddit(reddit_faiss, level)
        reddit_rows.append({'level': level, 'comment_count': count, 'n_queries': n_queries, 'recall_at_15': round(100.0 * recall, 1)})
    reddit_out = tables_dir / 'reddit_recall_at_15.csv'
    pd.DataFrame(reddit_rows).to_csv(reddit_out, index=False)
    combined_rows: list[dict[str, object]] = []
    for (level, count) in zip(LEVELS, COMMENT_COUNTS):
        (reddit_recall, reddit_n) = recall_at_15_reddit(reddit_faiss, level)
        (hn_recall, hn_n) = recall_at_15_hn(hn_faiss, level)
        reddit_top1 = REDDIT_TOP1[level]
        hn_top1 = HN_TOP1[level]
        reddit_recall_pct = 100.0 * reddit_recall
        hn_recall_pct = 100.0 * hn_recall
        combined_rows.append({'level': level, 'comment_count': count, 'reddit_recall_at_15': round(reddit_recall_pct, 1), 'reddit_top1_accuracy': reddit_top1, 'reddit_conditional_reason_accuracy': round(100.0 * reddit_top1 / reddit_recall_pct if reddit_recall_pct > 0 else 0.0, 1), 'hn_recall_at_15': round(hn_recall_pct, 1), 'hn_top1_accuracy': hn_top1, 'hn_conditional_reason_accuracy': round(100.0 * hn_top1 / hn_recall_pct if hn_recall_pct > 0 else 0.0, 1)})
    search_reason_out = tables_dir / 'search_vs_reason.csv'
    pd.DataFrame(combined_rows).to_csv(search_reason_out, index=False)
    print('Saved outputs:')
    print(f'- {reddit_out}')
    print(f'- {search_reason_out}')
    print('\nReddit Recall@15:')
    print(pd.DataFrame(reddit_rows).to_string(index=False))
    print('\nSearch vs Reason (both platforms):')
    print(pd.DataFrame(combined_rows).to_string(index=False))
if __name__ == '__main__':
    main()
