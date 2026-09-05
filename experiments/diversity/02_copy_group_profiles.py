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
    for group in GROUPS:
        out_dir = p['group_profiles_root'] / group
        out_dir.mkdir(parents=True, exist_ok=True)
    copied = {group: 0 for group in GROUPS}
    missing_t4 = 0
    for (_, row) in manifest.iterrows():
        user_id = str(row['user_id'])
        group = str(row['diversity_group'])
        src = p['t4_truncated'] / f'{user_id}.txt'
        dst = p['group_profiles_root'] / group / f'{user_id}.txt'
        if not src.exists():
            missing_t4 += 1
            continue
        shutil.copy2(src, dst)
        copied[group] += 1
    print('Copied T4 query profiles by group:')
    for group in GROUPS:
        print(f"- {group}: {copied[group]:,} files -> {p['group_profiles_root'] / group}")
    if missing_t4:
        print(f'Missing T4 profile files: {missing_t4:,}')
if __name__ == '__main__':
    main()
