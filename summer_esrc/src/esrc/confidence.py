"""P2 confidence estimators (a)/(b)/(c) over identical Reason picks."""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

from esrc.generate import TokenLogprob


def estimator_a_verbalized(confidence: float) -> float:
    """(a) Model's self-reported confidence in [0, 1]."""
    return max(0.0, min(1.0, float(confidence)))


def _join_tokens(tokens: Sequence[TokenLogprob]) -> str:
    return "".join(t.token for t in tokens)


def find_selected_number_span(
    token_logprobs: Sequence[TokenLogprob],
    selected_number: int,
) -> Optional[tuple[int, int]]:
    """
    Locate the token span for the selected candidate number inside the
    completion (prefer the value after selected_candidate_number).

    Returns [start, end) indices into token_logprobs, or None.
    """
    if not token_logprobs or selected_number < 1:
        return None

    pieces = [t.token for t in token_logprobs]
    full = "".join(pieces)
    target = str(selected_number)

    # Prefer the JSON field value.
    patterns = [
        rf'"selected_candidate_number"\s*:\s*({re.escape(target)})\b',
        rf"selected_candidate_number\s*[:=]\s*({re.escape(target)})\b",
    ]
    char_start = char_end = None
    for pat in patterns:
        m = re.search(pat, full)
        if m:
            char_start, char_end = m.start(1), m.end(1)
            break
    if char_start is None:
        # Fallback: last standalone occurrence of the number
        matches = list(re.finditer(rf"(?<!\d){re.escape(target)}(?!\d)", full))
        if not matches:
            return None
        m = matches[-1]
        char_start, char_end = m.start(), m.end()

    # Map character span → token indices
    pos = 0
    start_i = end_i = None
    for i, tok in enumerate(pieces):
        nxt = pos + len(tok)
        if start_i is None and nxt > char_start:
            start_i = i
        if start_i is not None and nxt >= char_end:
            end_i = i + 1
            break
        pos = nxt
    if start_i is None or end_i is None:
        return None
    return start_i, end_i


def estimator_b_selected_id_logprob(
    token_logprobs: Sequence[TokenLogprob],
    selected_number: int,
) -> Optional[float]:
    """
    (b) Sum of logprobs over tokens that name the selected candidate number
    only — not the full completion / reasoning prose.
    """
    span = find_selected_number_span(token_logprobs, selected_number)
    if span is None:
        return None
    start, end = span
    return float(sum(token_logprobs[i].logprob for i in range(start, end)))


def logprob_to_unit_interval(lp: float) -> float:
    """Map a (typically negative) log-prob to (0, 1] via exp, clamped."""
    # Guard overflow; very negative → ~0
    if lp > 0:
        lp = 0.0
    try:
        return float(math.exp(lp))
    except OverflowError:
        return 0.0


def softmax(xs: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not xs:
        return []
    t = float(temperature) if temperature else 1.0
    scaled = [x / t for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    if z <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in exps]


def estimator_c_softmax_mass_on_selected(
    candidate_scores: Sequence[float],
    selected_index_0based: int,
    temperature: float = 1.0,
) -> tuple[float, list[float]]:
    """
    (c) Softmax over per-candidate scores; return mass on the P2.3-selected
    index (same pick as a/b). Also returns the full probability vector.
    """
    probs = softmax(candidate_scores, temperature=temperature)
    if not (0 <= selected_index_0based < len(probs)):
        raise IndexError(
            f"selected_index_0based={selected_index_0based} not in [0, {len(probs)})"
        )
    return float(probs[selected_index_0based]), probs


def _norm_tok(t: str) -> str:
    return t.strip().upper()


def first_token_option_logprobs(
    token_logprobs: Sequence[TokenLogprob],
    options: Sequence[str],
) -> dict[str, float]:
    """
    From the first non-empty completion token's top_logprobs, collect the best
    logprob for each option name (case/whitespace-insensitive).
    """
    wanted = {_norm_tok(o): o for o in options}
    found: dict[str, float] = {}
    for tok in token_logprobs:
        cands: list[tuple[str, float]] = [(tok.token, tok.logprob)]
        cands.extend(list(tok.top_logprobs or ()))
        if not any(_norm_tok(t) for t, _ in cands):
            continue  # skip blank / special first tokens
        for raw, lp in cands:
            key = _norm_tok(raw)
            if key in wanted:
                canon = wanted[key]
                prev = found.get(canon)
                if prev is None or lp > prev:
                    found[canon] = float(lp)
        if found:
            break
    return found


def binary_choice_probability(
    token_logprobs: Sequence[TokenLogprob],
    positive: str,
    negative: str,
) -> Optional[tuple[float, float, float]]:
    """
    P(positive) via 2-way softmax over positive/negative token logprobs
    from the first completion token. Returns (p_pos, lp_pos, lp_neg) or None.
    """
    found = first_token_option_logprobs(token_logprobs, [positive, negative])
    if positive not in found or negative not in found:
        return None
    lp_pos = found[positive]
    lp_neg = found[negative]
    # 2-class softmax
    m = max(lp_pos, lp_neg)
    e_pos = math.exp(lp_pos - m)
    e_neg = math.exp(lp_neg - m)
    z = e_pos + e_neg
    return float(e_pos / z), float(lp_pos), float(lp_neg)
