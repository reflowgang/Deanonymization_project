from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBED_MODEL, paths, type_folders

def list_summary_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted((p for p in dir_path.glob('*.txt') if p.is_file()))

def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8').strip()

def common_summary_user_ids(summaries_root: Path) -> set[str]:
    common: set[str] | None = None
    for folder in type_folders():
        summary_dir = summaries_root / folder
        user_ids = {path.stem for path in summary_dir.glob('*.txt') if path.is_file() and path.stat().st_size > 0}
        common = user_ids if common is None else common & user_ids
    return common or set()

def main() -> None:
    p = paths()
    p['type_embeddings_root'].mkdir(parents=True, exist_ok=True)
    cohort = common_summary_user_ids(p['type_summaries_root'])
    if not cohort:
        raise RuntimeError('No users with non-empty summaries for all three content types.')
    print(f'ESRC cohort (all three summaries): {len(cohort):,} users')
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:
        raise RuntimeError('sentence-transformers is required. Install with: pip install sentence-transformers') from e
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = None
    print(f'Model: {EMBED_MODEL}')
    model = SentenceTransformer(EMBED_MODEL)
    for folder in type_folders():
        summary_dir = p['type_summaries_root'] / folder
        files = list_summary_files(summary_dir)
        if not files:
            raise FileNotFoundError(f"No summaries found for type '{folder}' in {summary_dir}")
        texts: list[str] = []
        user_ids: list[str] = []
        summary_paths: list[str] = []
        it = files
        if tqdm is not None:
            it = tqdm(files, desc=f'load {folder}')
        for file_path in it:
            if file_path.stem not in cohort:
                continue
            text = read_text(file_path)
            if not text:
                continue
            texts.append(text)
            user_ids.append(file_path.stem)
            summary_paths.append(str(file_path))
        if not texts:
            raise ValueError(f"Type '{folder}' has no non-empty summaries.")
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=tqdm is not None, normalize_embeddings=False).astype(np.float32, copy=False)
        type_emb_dir = p['type_embeddings_root'] / folder
        type_emb_dir.mkdir(parents=True, exist_ok=True)
        out_npy = type_emb_dir / 'query_embeddings.npy'
        np.save(out_npy, embeddings)
        index_df = pd.DataFrame({'row_id': list(range(len(user_ids))), 'user_id': user_ids, 'content_type': [folder] * len(user_ids), 'summary_path': summary_paths})
        out_index_csv = type_emb_dir / 'query_embeddings_index.csv'
        index_df.to_csv(out_index_csv, index=False)
        print(f'- {folder}: embedded {len(user_ids):,} summaries -> {out_npy}')
    print('\nDone.')
if __name__ == '__main__':
    main()
