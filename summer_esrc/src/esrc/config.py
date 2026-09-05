from __future__ import annotations
from pathlib import Path
from typing import Optional

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_dotenv_files(root: Optional[Path]=None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = root or project_root()
    for name in ('.env', 'api_key.env'):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)
