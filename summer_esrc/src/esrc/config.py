"""Lightweight path / env helpers for the shared pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def project_root() -> Path:
    # src/esrc/config.py -> src/esrc -> src -> project root
    return Path(__file__).resolve().parents[2]


def load_dotenv_files(root: Optional[Path] = None) -> None:
    """Load .env then api_key.env if present (does not override existing env)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = root or project_root()
    for name in (".env", "api_key.env"):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)
