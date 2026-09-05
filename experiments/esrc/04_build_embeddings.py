from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
LEVELS_ALL: list[str] = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

@dataclass(frozen=True)
class Paths:
    summaries_root: Path
    embeddings_dir: Path
    results_tables_dir: Path

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def list_summary_files(level_dir: Path) -> list[Path]:
    if not level_dir.exists():
        return []
    return sorted((p for p in level_dir.glob('*.txt') if p.is_file()))

def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8').strip()

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = Paths(summaries_root=project_root / 'data/esrc/pool_en/summaries', embeddings_dir=project_root / 'data/esrc/pool_en/embeddings', results_tables_dir=project_root / 'results/tables')
    _ensure_dir(paths.embeddings_dir)
    _ensure_dir(paths.results_tables_dir)
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:
        raise RuntimeError('sentence-transformers is required but not installed. Install it, e.g.\n  pip install sentence-transformers\nThen rerun this script.') from e
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = None
    print(f'Input summaries root: {paths.summaries_root}')
    print(f'Output embeddings dir: {paths.embeddings_dir}')
    print(f'Model: {MODEL_NAME}')
    model = SentenceTransformer(MODEL_NAME)
    for level in LEVELS_ALL:
        level_dir = paths.summaries_root / level
        files = list_summary_files(level_dir)
        if not files:
            print(f'- {level}: 0 summaries (skipping)')
            continue
        texts: list[str] = []
        user_ids: list[str] = []
        summary_paths: list[str] = []
        it = files
        if tqdm is not None:
            it = tqdm(files, desc=f'load {level}')
        for p in it:
            text = read_text(p)
            if not text:
                continue
            texts.append(text)
            user_ids.append(p.stem)
            summary_paths.append(str(p))
        if not texts:
            print(f'- {level}: 0 non-empty summaries (skipping)')
            continue
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=tqdm is not None, normalize_embeddings=False).astype(np.float32, copy=False)
        out_npy = paths.embeddings_dir / f'query_embeddings_{level}.npy'
        np.save(out_npy, embeddings)
        index_df = pd.DataFrame({'row_id': list(range(len(user_ids))), 'user_id': user_ids, 'T_level': [level] * len(user_ids), 'summary_path': summary_paths})
        out_index_csv = paths.results_tables_dir / f'pool_en_query_embeddings_index_{level}.csv'
        index_df.to_csv(out_index_csv, index=False)
        print(f'- {level}: embedded {len(user_ids):,} summaries -> {out_npy.name}')
    print('\nDone.')
if __name__ == '__main__':
    main()
