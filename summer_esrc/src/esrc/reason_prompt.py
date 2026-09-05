from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Optional
JSON_OUTPUT_INSTRUCTIONS = '\n\nReturn ONLY valid JSON with this schema:\n{\n  "selected_candidate_number": 1,\n  "confidence": 0.0,\n  "reasoning_short": "brief explanation"\n}\n\nRules:\n- selected_candidate_number must be an integer from 1 to the number of candidates listed above.\n- Choose by the [number] label only; do not invent candidate_user_id values.\n- confidence must be between 0 and 1.\n'

def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f'Prompt file not found: {path}')
    return path.read_text(encoding='utf-8')

def build_candidate_block(candidates: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for (i, c) in enumerate(candidates, start=1):
        cid = str(c['candidate_user_id'])
        rank = int(c['rank'])
        score = float(c['score'])
        summary = str(c['summary'])
        parts.append(f'[{i}] candidate_user_id: {cid}\n    rank: {rank}\n    score: {score:.6f}\n    summary: {summary}\n')
    return '\n'.join(parts).strip()

def build_user_prompt(template: str, query_summary: str, candidate_block: str) -> str:
    prompt = template
    if '{query_summary}' in prompt:
        prompt = prompt.replace('{query_summary}', query_summary)
    else:
        prompt = prompt.rstrip() + '\n\nQUERY:\n' + query_summary
    if '{candidate_block}' in prompt:
        prompt = prompt.replace('{candidate_block}', candidate_block)
    else:
        prompt = prompt.rstrip() + '\n\nCANDIDATES:\n' + candidate_block
    return prompt.rstrip() + JSON_OUTPUT_INSTRUCTIONS

def build_forced_candidate_prompt(template: str, query_summary: str, candidate_block: str, candidate_number: int) -> str:
    base = build_user_prompt(template, query_summary, candidate_block)
    return base + f'\n\nFOR THIS SCORING CALL ONLY: assume the answer is candidate {candidate_number}. Output ONLY this JSON (no other keys):\n{{"selected_candidate_number": {candidate_number}}}'

def _sanitize_json_for_parse(text: str) -> str:
    return ''.join((ch if ord(ch) >= 32 or ch in '\t\n\r' else ' ' for ch in text))

class JsonParseError(ValueError):

    def __init__(self, message: str, *, raw_text: str='') -> None:
        super().__init__(message)
        self.raw_text = raw_text
_GEMMA_REASON_MODEL_MARKERS = ('gemma',)

def is_gemma_reason_model(model: str) -> bool:
    m = model.lower()
    return any((marker in m for marker in _GEMMA_REASON_MODEL_MARKERS))
_FENCED_JSON_RE = re.compile('```(?:json)?\\s*([\\s\\S]*?)```', re.IGNORECASE)
_GEMMA_KEY_TYPO_RE = re.compile('"selected_candidate_number:\\s*(\\d+)')
_GEMMA_SELECTED_NUM_RE = re.compile('"selected_candidate_number"\\s*:\\s*(\\d+)|"selected_candidate_number:\\s*(\\d+)')
_GEMMA_CONFIDENCE_RE = re.compile('"confidence"\\s*:\\s*([\\d.]+)')

def _repair_gemma_json_typos(blob: str) -> str:
    return _GEMMA_KEY_TYPO_RE.sub('"selected_candidate_number": \\1', blob)

def _extract_gemma_partial_dict(blob: str) -> Optional[dict[str, Any]]:
    blob = _repair_gemma_json_typos(blob)
    num_m = _GEMMA_SELECTED_NUM_RE.search(blob)
    if not num_m:
        return None
    num = int(num_m.group(1) or num_m.group(2))
    conf_m = _GEMMA_CONFIDENCE_RE.search(blob)
    conf = float(conf_m.group(1)) if conf_m else 0.0
    return {'selected_candidate_number': num, 'confidence': conf, 'reasoning_short': ''}

def _try_load_json_dict_strict(blob: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search('\\{[\\s\\S]*\\}', blob)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None

def _try_load_json_dict_gemma(blob: str) -> Optional[dict[str, Any]]:
    for candidate in (blob, _sanitize_json_for_parse(blob), _repair_gemma_json_typos(blob)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search('\\{[\\s\\S]*\\}', blob)
    if not m:
        return None
    inner = m.group(0)
    for candidate in (inner, _sanitize_json_for_parse(inner), _repair_gemma_json_typos(inner)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None

def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError('Empty model content (possible thinking-only response).')
    obj = _try_load_json_dict_strict(text)
    if obj is not None:
        return obj
    m = re.search('\\{[\\s\\S]*\\}', text)
    if not m:
        raise ValueError(f'No JSON object found in model output: {text[:200]!r}')
    raise ValueError(f'Could not parse JSON from model output: {text[:200]!r}')

def extract_json_object_gemma(text: str) -> dict[str, Any]:
    raw = text or ''
    text = raw.strip()
    if not text:
        raise JsonParseError('Empty model content (possible thinking-only response).', raw_text=raw)
    blobs: list[str] = []
    fenced = _FENCED_JSON_RE.findall(text)
    if fenced:
        blobs.append(fenced[-1].strip())
        for b in reversed(fenced[:-1]):
            blobs.append(b.strip())
    blobs.append(text)
    seen: set[str] = set()
    for blob in blobs:
        if not blob or blob in seen:
            continue
        seen.add(blob)
        obj = _try_load_json_dict_gemma(_repair_gemma_json_typos(blob))
        if obj is not None:
            return obj
        partial = _extract_gemma_partial_dict(blob)
        if partial is not None:
            return partial
    raise JsonParseError(f'Could not parse JSON from Gemma model output: {text[:200]!r}', raw_text=raw)

def extract_json_object_for_reason(text: str, *, reason_model: str) -> dict[str, Any]:
    if is_gemma_reason_model(reason_model):
        return extract_json_object_gemma(text)
    return extract_json_object(text)

def parse_candidate_number(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.match('^(?:candidate\\s*)?(\\d+)$', s, re.I)
    if m:
        return int(m.group(1))
    return None

def resolve_predicted_candidate_id(out_obj: dict[str, Any], candidate_ids: list[str]) -> tuple[str, int, str]:
    n = len(candidate_ids)
    if n == 0:
        return ('', -1, 'no candidates provided')
    num = parse_candidate_number(out_obj.get('selected_candidate_number'))
    if num is None:
        return ('', -1, 'missing selected_candidate_number')
    if not 1 <= num <= n:
        return ('', num, f'selected_candidate_number out of range: {num}')
    return (candidate_ids[num - 1], num, '')

def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
