"""Single LLM generate() wrapper for local vLLM (OpenAI-compatible API)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import APIStatusError, BadRequestError, OpenAI


# Served limits on gpu6.sedan.pro (from /v1/models). Keep Extract vs Reason distinct.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "qwen3.5-4b": 16384,
    "qwen3.6-35b-a3b-nvfp4": 32768,
    "gemma4-26b": 65536,
}
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "qwen3.5-4b": 8192,
    "qwen3.6-35b-a3b-nvfp4": 8192,
    "gemma4-26b": 8192,
}

# Default: no SDK retries. Blind resends of 400/context-exceeded flood the
# shared queue; callers that want transient retries must opt in explicitly.
DEFAULT_MAX_RETRIES = 0


class PermanentRequestError(RuntimeError):
    """Non-retryable request failure. Do not resend the same payload unchanged."""

    def __init__(self, message: str, *, model: str = "", max_tokens: int | None = None):
        super().__init__(message)
        self.model = model
        self.max_tokens = max_tokens


class ContextLengthExceededError(PermanentRequestError):
    """Permanent failure: prompt + max_tokens exceeds model context."""


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float
    top_logprobs: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    finish_reason: Optional[str]
    # Sum of chosen-token logprobs over the completion (None if logprobs disabled / missing).
    sequence_logprob: Optional[float]
    token_logprobs: tuple[TokenLogprob, ...] = ()
    reasoning_content: Optional[str] = None
    raw: Any = field(default=None, repr=False)


def is_context_overflow_message(msg: str) -> bool:
    msg_l = msg.lower()
    return (
        "maximum context length" in msg_l
        or ("context length" in msg_l and "exceed" in msg_l)
        or "prompt contains at least" in msg_l
        or "your prompt contains" in msg_l
        or ("context length" in msg_l and "tokens" in msg_l)
        or "max_model_len" in msg_l
        or "max sequence length" in msg_l
    )


def is_http_400(exc: BaseException) -> bool:
    if isinstance(exc, BadRequestError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, APIStatusError) and status == 400


def is_context_overflow_error(exc: BaseException) -> bool:
    """True when the failure is a context-length overflow (safe for Extract split)."""
    if isinstance(exc, ContextLengthExceededError):
        return True
    if isinstance(exc, PermanentRequestError):
        return False
    return is_context_overflow_message(str(exc))


def context_limit_for_model(model: str) -> Optional[int]:
    return MODEL_CONTEXT_LIMITS.get(model)


def clamp_max_tokens(
    model: str,
    max_tokens: int,
    *,
    prompt_tokens: Optional[int] = None,
) -> int:
    """Cap max_tokens by served max_output and remaining context headroom.

    Never silently raise max_tokens toward Extract's 16k limit on Reason calls.
    If prompt already fills the context, raise ContextLengthExceededError.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

    out_cap = MODEL_MAX_OUTPUT_TOKENS.get(model)
    capped = min(max_tokens, out_cap) if out_cap is not None else max_tokens

    ctx = context_limit_for_model(model)
    if ctx is not None and prompt_tokens is not None:
        room = ctx - int(prompt_tokens) - 1
        if room <= 0:
            raise ContextLengthExceededError(
                f"Prompt already exceeds context for {model}: "
                f"prompt_tokens={prompt_tokens} context_limit={ctx}",
                model=model,
                max_tokens=max_tokens,
            )
        capped = min(capped, room)

    return capped


def get_client(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """Build an OpenAI client pointed at vLLM (or any OpenAI-compatible server).

    max_retries defaults to 0 so 400/context-exceeded (and litellm 500 wraps)
    are never blindly resent in a loop by the SDK.
    """
    resolved_base = (base_url or os.getenv("VLLM_BASE_URL") or "").rstrip("/")
    if not resolved_base:
        raise RuntimeError(
            "VLLM_BASE_URL is not set. Example: export VLLM_BASE_URL=http://127.0.0.1:8000/v1"
        )
    resolved_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
    return OpenAI(
        base_url=resolved_base,
        api_key=resolved_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def list_models(client: Optional[OpenAI] = None) -> list[str]:
    client = client or get_client()
    page = client.models.list()
    return [m.id for m in page.data]


def _parse_token_logprobs(choice: Any) -> tuple[TokenLogprob, ...]:
    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None:
        return ()
    content = getattr(logprobs_obj, "content", None)
    if not content:
        return ()

    parsed: list[TokenLogprob] = []
    for item in content:
        top_raw = getattr(item, "top_logprobs", None) or []
        top: list[tuple[str, float]] = []
        for t in top_raw:
            top.append((str(getattr(t, "token", "")), float(getattr(t, "logprob", 0.0))))
        parsed.append(
            TokenLogprob(
                token=str(getattr(item, "token", "")),
                logprob=float(getattr(item, "logprob", 0.0)),
                top_logprobs=tuple(top),
            )
        )
    return tuple(parsed)


def _message_reasoning(message: Any) -> Optional[str]:
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val:
            return str(val)
    return None


def generate(
    messages: list[dict[str, str]],
    *,
    model: str,
    client: Optional[OpenAI] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    enable_thinking: Optional[bool] = False,
    extra: Optional[dict[str, Any]] = None,
    prompt_tokens: Optional[int] = None,
) -> GenerationResult:
    """
    One shared generation entrypoint for Extract/Reason (and P2 estimators).

    enable_thinking: when False (default), requests
    extra_body.chat_template_kwargs.enable_thinking=false for Qwen-style
    thinking templates on vLLM. Pass None to leave server defaults unchanged.

    On 400 / context-exceeded: raises ContextLengthExceededError (permanent).
    Does not retry that request.
    """
    client = client or get_client()
    safe_max = clamp_max_tokens(model, max_tokens, prompt_tokens=prompt_tokens)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": safe_max,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None:
            kwargs["top_logprobs"] = int(top_logprobs)

    extra_body: dict[str, Any] = {}
    if enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    if extra:
        # Allow callers to merge/override extra_body
        caller_extra = dict(extra)
        caller_body = caller_extra.pop("extra_body", None)
        if isinstance(caller_body, dict):
            # shallow merge; nested chat_template_kwargs: caller wins on keys
            merged = dict(extra_body)
            for k, v in caller_body.items():
                if k == "chat_template_kwargs" and isinstance(v, dict):
                    base = dict(merged.get("chat_template_kwargs") or {})
                    base.update(v)
                    merged["chat_template_kwargs"] = base
                else:
                    merged[k] = v
            extra_body = merged
        kwargs.update(caller_extra)
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        # Context overflow → permanent, never resend as-is.
        if is_context_overflow_message(str(exc)):
            raise ContextLengthExceededError(
                f"Context exceeded for model={model} max_tokens={safe_max}: {exc}",
                model=model,
                max_tokens=safe_max,
            ) from exc
        # Other 400s are also non-retryable (SDK max_retries=0 + resume skip).
        if is_http_400(exc):
            raise PermanentRequestError(
                f"Non-retryable HTTP 400 for model={model} max_tokens={safe_max}: {exc}",
                model=model,
                max_tokens=safe_max,
            ) from exc
        raise

    choice = response.choices[0]
    message = choice.message
    text = (message.content or "").strip()
    reasoning = _message_reasoning(message)
    token_lps = _parse_token_logprobs(choice) if logprobs else ()
    seq_lp: Optional[float] = None
    if token_lps:
        seq_lp = float(sum(t.logprob for t in token_lps))

    return GenerationResult(
        text=text,
        model=getattr(response, "model", None) or model,
        finish_reason=getattr(choice, "finish_reason", None),
        sequence_logprob=seq_lp,
        token_logprobs=token_lps,
        reasoning_content=reasoning,
        raw=response,
    )
