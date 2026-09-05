from __future__ import annotations
import json
from pathlib import Path
PERMANENT_STATUSES = frozenset({'permanent_error'})
PERMANENT_ERROR_TYPES = frozenset({'ContextLengthExceededError', 'PermanentRequestError', 'context_length_exceeded', 'BadRequestError'})

def row_user_id(obj: dict) -> str | None:
    uid = obj.get('query_user_id') or obj.get('user_id')
    if uid is None:
        return None
    return str(uid)

def is_permanent_failure_row(obj: dict) -> bool:
    if obj.get('status') in PERMANENT_STATUSES:
        return True
    err_type = str(obj.get('error_type') or '')
    if err_type in PERMANENT_ERROR_TYPES:
        return True
    err = str(obj.get('error') or '')
    if 'ContextLengthExceededError' in err or 'context_length_exceeded' in err.lower():
        return True
    if 'maximum context length' in err.lower():
        return True
    return False

def load_ok_user_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('status') != 'ok':
                continue
            uid = row_user_id(obj)
            if uid:
                done.add(uid)
    return done

def load_resume_skip_user_ids(path: Path) -> set[str]:
    last: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                uid = row_user_id(obj)
                if uid:
                    last[uid] = obj
    skip: set[str] = set()
    for (uid, obj) in last.items():
        if obj.get('status') == 'ok' or is_permanent_failure_row(obj):
            skip.add(uid)
    return skip

def load_last_ok_by_user(path: Path) -> dict[str, dict]:
    last: dict[str, dict] = {}
    if not path.exists():
        return last
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('status') != 'ok':
                continue
            uid = row_user_id(obj)
            if uid:
                last[uid] = obj
    return last
