"""Reason-stage prompt building (Lermen record selection) + JSON parse helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


JSON_OUTPUT_INSTRUCTIONS = """

Return ONLY valid JSON with this schema:
{
  "selected_candidate_number": 1,
  "confidence": 0.0,
  "reasoning_short": "brief explanation"
}

Rules:
- selected_candidate_number must be an integer from 1 to the number of candidates listed above.
- Choose by the [number] label only; do not invent candidate_user_id values.
- confidence must be between 0 and 1.
"""


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_candidate_block(candidates: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(candidates, start=1):
        cid = str(c["candidate_user_id"])
        rank = int(c["rank"])
        score = float(c["score"])
        summary = str(c["summary"])
        parts.append(
            f"[{i}] candidate_user_id: {cid}\n"
            f"    rank: {rank}\n"
            f"    score: {score:.6f}\n"
            f"    summary: {summary}\n"
        )
    return "\n".join(parts).strip()


def build_user_prompt(template: str, query_summary: str, candidate_block: str) -> str:
    prompt = template
    if "{query_summary}" in prompt:
        prompt = prompt.replace("{query_summary}", query_summary)
    else:
        prompt = prompt.rstrip() + "\n\nQUERY:\n" + query_summary

    if "{candidate_block}" in prompt:
        prompt = prompt.replace("{candidate_block}", candidate_block)
    else:
        prompt = prompt.rstrip() + "\n\nCANDIDATES:\n" + candidate_block

    return prompt.rstrip() + JSON_OUTPUT_INSTRUCTIONS


def build_forced_candidate_prompt(
    template: str,
    query_summary: str,
    candidate_block: str,
    candidate_number: int,
) -> str:
    """
    Prompt for estimator (c): same evidence, ask for logprob of asserting
    that a specific candidate number is the match (forced hypothesis).
    """
    base = build_user_prompt(template, query_summary, candidate_block)
    return (
        base
        + f"\n\nFOR THIS SCORING CALL ONLY: assume the answer is candidate "
        f"{candidate_number}. Output ONLY this JSON "
        f'(no other keys):\n{{"selected_candidate_number": {candidate_number}}}'
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model content (possible thinking-only response).")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Model returned non-object JSON.")
    return obj


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
    m = re.match(r"^(?:candidate\s*)?(\d+)$", s, re.I)
    if m:
        return int(m.group(1))
    return None


def resolve_predicted_candidate_id(
    out_obj: dict[str, Any],
    candidate_ids: list[str],
) -> tuple[str, int, str]:
    """Returns (candidate_user_id, 1-based number, error)."""
    n = len(candidate_ids)
    if n == 0:
        return "", -1, "no candidates provided"
    num = parse_candidate_number(out_obj.get("selected_candidate_number"))
    if num is None:
        return "", -1, "missing selected_candidate_number"
    if not (1 <= num <= n):
        return "", num, f"selected_candidate_number out of range: {num}"
    return candidate_ids[num - 1], num, ""


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
