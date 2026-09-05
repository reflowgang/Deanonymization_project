from __future__ import annotations
import shutil
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GROUPS, paths

def main() -> None:
    p = paths()
    manifest = pd.read_csv(p['group_manifest'])
    linked = {group: 0 for group in GROUPS}
    missing_main = 0
    for group in GROUPS:
        out_dir = p['group_summaries_root'] / group
        out_dir.mkdir(parents=True, exist_ok=True)
    for (_, row) in manifest.iterrows():
        user_id = str(row['user_id'])
        group = str(row['diversity_group'])
        src = p['main_t4_summaries'] / f'{user_id}.txt'
        dst = p['group_summaries_root'] / group / f'{user_id}.txt'
        if not src.exists() or src.stat().st_size == 0:
            missing_main += 1
            continue
        shutil.copy2(src, dst)
        linked[group] += 1
    print('Linked main T4 summaries into diversity group folders:')
    for group in GROUPS:
        print(f"- {group}: {linked[group]:,} -> {p['group_summaries_root'] / group}")
    if missing_main:
        print(f'Missing or empty main T4 summaries: {missing_main:,}')
    print(f'Total linked: {sum(linked.values()):,} / {len(manifest):,}')
    if sum(linked.values()) != len(manifest):
        raise RuntimeError('Not all users received summaries. Run 03_extract_summaries.py for gaps.')
if __name__ == '__main__':
    main()
