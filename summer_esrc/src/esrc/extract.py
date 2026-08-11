"""Conditional Extract with token-probe gating and chunk-and-merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from esrc.generate import (
    MODEL_CONTEXT_LIMITS,
    ContextLengthExceededError,
    clamp_max_tokens,
    generate,
    is_context_overflow_error,
    is_context_overflow_message,
)

# Defaults for Extract on qwen3.5-4b; overridden per-model via context_limit_for_model().
CONTEXT_LIMIT_TOKENS = 16384
CHUNK_GATE_TOKENS = 14_000
PROBE_MAX_TOKENS = 1
DEFAULT_EXTRACT_MAX_TOKENS = 1024
_GATE_MARGIN = 2_000  # leave headroom for output + template drift


def context_limit_for_model(model: str) -> int:
    return int(MODEL_CONTEXT_LIMITS.get(model, CONTEXT_LIMIT_TOKENS))


def chunk_gate_for_model(model: str) -> int:
    return max(1_000, context_limit_for_model(model) - _GATE_MARGIN)


def is_context_overflow(msg: str) -> bool:
    """Back-compat alias used by Extract chunking fallback."""
    return is_context_overflow_message(msg)


@dataclass(frozen=True)
class ExtractResult:
    summary: str
    chunked: bool
    n_chunks: int
    method: str  # single | planned_chunks | recursive_fallback
    chunk_sizes: tuple[int, ...] = ()
    probe_tokens_full: Optional[int] = None
    probe_tokens_per_chunk: tuple[int, ...] = ()


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_user_content(prompt_template: str, profile_text: str) -> str:
    if "{profile_text}" in prompt_template:
        return prompt_template.replace("{profile_text}", profile_text)
    return prompt_template.rstrip() + "\n\nCOMMENTS:\n" + profile_text


def build_merge_content(merge_template: str, chunk_summaries: list[str]) -> str:
    numbered = "\n\n".join(
        f"[{i}] {s.strip()}" for i, s in enumerate(chunk_summaries, start=1) if s.strip()
    )
    if "{chunk_summaries}" in merge_template:
        return merge_template.replace("{chunk_summaries}", numbered)
    return merge_template.rstrip() + "\n\n" + numbered


def split_comments(profile_text: str) -> list[str]:
    lines = [ln.strip() for ln in profile_text.splitlines()]
    return [ln for ln in lines if ln]


def join_comments(comments: list[str]) -> str:
    return "\n".join(comments)


def measure_prompt_tokens(
    client: OpenAI,
    *,
    model: str,
    user_content: str,
    enable_thinking: bool = False,
) -> tuple[str, Optional[int]]:
    """Cheap probe: max_tokens=1, read usage.prompt_tokens."""
    try:
        safe_max = clamp_max_tokens(model, PROBE_MAX_TOKENS)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=safe_max,
            messages=[{"role": "user", "content": user_content}],
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
        )
        usage = getattr(resp, "usage", None)
        if usage and usage.prompt_tokens is not None:
            return "ok", int(usage.prompt_tokens)
        return "ok", None
    except Exception as e:  # noqa: BLE001
        if is_context_overflow_error(e) or is_context_overflow(str(e)):
            return "exceeds_limit", None
        raise

def _needs_chunking(
    client: OpenAI,
    *,
    model: str,
    extract_template: str,
    comments: list[str],
    enable_thinking: bool,
    chunk_gate: int,
) -> tuple[bool, Optional[int]]:
    content = build_user_content(extract_template, join_comments(comments))
    status, tokens = measure_prompt_tokens(
        client, model=model, user_content=content, enable_thinking=enable_thinking
    )
    if status == "exceeds_limit":
        return True, None
    if tokens is None:
        return False, None
    return tokens > chunk_gate, tokens


def plan_comment_chunks(
    comments: list[str],
    *,
    client: OpenAI,
    model: str,
    extract_template: str,
    enable_thinking: bool = False,
    chunk_gate: Optional[int] = None,
) -> tuple[list[list[str]], Optional[int]]:
    """Recursively split comment lists until each chunk's probe <= chunk_gate."""
    if chunk_gate is None:
        chunk_gate = chunk_gate_for_model(model)
    if not comments:
        return [[]], 0

    needs, full_tokens = _needs_chunking(
        client,
        model=model,
        extract_template=extract_template,
        comments=comments,
        enable_thinking=enable_thinking,
        chunk_gate=chunk_gate,
    )
    if not needs:
        return [comments], full_tokens

    if len(comments) <= 1:
        raise RuntimeError(
            "Single comment exceeds Extract context limit; cannot split further."
        )

    mid = len(comments) // 2
    left, _ = plan_comment_chunks(
        comments[:mid],
        client=client,
        model=model,
        extract_template=extract_template,
        enable_thinking=enable_thinking,
        chunk_gate=chunk_gate,
    )
    right, _ = plan_comment_chunks(
        comments[mid:],
        client=client,
        model=model,
        extract_template=extract_template,
        enable_thinking=enable_thinking,
        chunk_gate=chunk_gate,
    )
    return left + right, full_tokens


def _summarize_chunk(
    client: OpenAI,
    *,
    model: str,
    extract_template: str,
    comments: list[str],
    max_tokens: int,
    enable_thinking: bool,
    seed: Optional[int],
) -> str:
    content = build_user_content(extract_template, join_comments(comments))
    result = generate(
        [{"role": "user", "content": content}],
        model=model,
        client=client,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        enable_thinking=enable_thinking,
    )
    if not result.text and result.reasoning_content:
        raise RuntimeError("Extract returned empty content (possible thinking leak).")
    return result.text.strip()


def _summarize_chunk_with_fallback(
    client: OpenAI,
    *,
    model: str,
    extract_template: str,
    merge_template: str,
    comments: list[str],
    max_tokens: int,
    enable_thinking: bool,
    seed: Optional[int],
) -> tuple[str, bool]:
    """Summarize comments; on overflow split in half and merge recursively."""
    try:
        text = _summarize_chunk(
            client,
            model=model,
            extract_template=extract_template,
            comments=comments,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        return text, False
    except Exception as e:  # noqa: BLE001
        # Split-and-merge is the ONE intentional retry path, and only with a
        # smaller payload — never resend the same overflowing request.
        if not (is_context_overflow_error(e) or is_context_overflow(str(e))):
            raise
        if len(comments) <= 1:
            raise ContextLengthExceededError(
                "Single comment exceeds context on Extract call.",
                model=model,
                max_tokens=max_tokens,
            ) from e
        mid = len(comments) // 2
        left, left_fb = _summarize_chunk_with_fallback(
            client,
            model=model,
            extract_template=extract_template,
            merge_template=merge_template,
            comments=comments[:mid],
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        right, right_fb = _summarize_chunk_with_fallback(
            client,
            model=model,
            extract_template=extract_template,
            merge_template=merge_template,
            comments=comments[mid:],
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        merged = _merge_summaries(
            client,
            model=model,
            merge_template=merge_template,
            chunk_summaries=[left, right],
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        return merged, True or left_fb or right_fb


def _merge_summaries(
    client: OpenAI,
    *,
    model: str,
    merge_template: str,
    chunk_summaries: list[str],
    max_tokens: int,
    enable_thinking: bool,
    seed: Optional[int],
) -> str:
    if len(chunk_summaries) == 1:
        return chunk_summaries[0].strip()
    content = build_merge_content(merge_template, chunk_summaries)
    result = generate(
        [{"role": "user", "content": content}],
        model=model,
        client=client,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        enable_thinking=enable_thinking,
    )
    return (result.text or "").strip()


def extract_profile(
    profile_text: str,
    *,
    client: OpenAI,
    model: str,
    extract_template: str,
    merge_template: str,
    max_tokens: int = DEFAULT_EXTRACT_MAX_TOKENS,
    enable_thinking: bool = False,
    seed: Optional[int] = None,
) -> ExtractResult:
    """
    Extract a trait summary from raw comment text.

    Gate: real vLLM prompt_tokens probe (max_tokens=1). Chunk when probe
    exceeds CHUNK_GATE_TOKENS or reports context overflow. Fallback: if a
    chunk Extract call overflows, split that chunk in half and merge.
    """
    comments = split_comments(profile_text)
    if not comments:
        return ExtractResult(
            summary="",
            chunked=False,
            n_chunks=0,
            method="empty",
            chunk_sizes=(),
        )

    chunk_lists, full_probe = plan_comment_chunks(
        comments,
        client=client,
        model=model,
        extract_template=extract_template,
        enable_thinking=enable_thinking,
    )
    chunked = len(chunk_lists) > 1
    method = "single" if not chunked else "planned_chunks"
    used_fallback = False

    chunk_summaries: list[str] = []
    probe_per_chunk: list[int] = []
    for chunk in chunk_lists:
        content = build_user_content(extract_template, join_comments(chunk))
        st, pt = measure_prompt_tokens(
            client, model=model, user_content=content, enable_thinking=enable_thinking
        )
        probe_per_chunk.append(pt if pt is not None else -1)

        summary, fb = _summarize_chunk_with_fallback(
            client,
            model=model,
            extract_template=extract_template,
            merge_template=merge_template,
            comments=chunk,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            seed=seed,
        )
        chunk_summaries.append(summary)
        used_fallback = used_fallback or fb

    if used_fallback:
        method = "recursive_fallback"

    final = _merge_summaries(
        client,
        model=model,
        merge_template=merge_template,
        chunk_summaries=chunk_summaries,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        seed=seed,
    )

    return ExtractResult(
        summary=final,
        chunked=chunked or used_fallback,
        n_chunks=len(chunk_lists),
        method=method,
        chunk_sizes=tuple(len(c) for c in chunk_lists),
        probe_tokens_full=full_probe,
        probe_tokens_per_chunk=tuple(probe_per_chunk),
    )
