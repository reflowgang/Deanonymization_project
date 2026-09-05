from __future__ import annotations
import math
import re
from typing import Optional, Sequence
from esrc.generate import TokenLogprob

def estimator_a_verbalized(confidence: float) -> float:
    return max(0.0, min(1.0, float(confidence)))

def _join_tokens(tokens: Sequence[TokenLogprob]) -> str:
    return ''.join((t.token for t in tokens))

def find_selected_number_span(token_logprobs: Sequence[TokenLogprob], selected_number: int) -> Optional[tuple[int, int]]:
    if not token_logprobs or selected_number < 1:
        return None
    pieces = [t.token for t in token_logprobs]
    full = ''.join(pieces)
    target = str(selected_number)
    patterns = [f'"selected_candidate_number"\\s*:\\s*({re.escape(target)})\\b', f'selected_candidate_number\\s*[:=]\\s*({re.escape(target)})\\b']
    char_start = char_end = None
    for pat in patterns:
        m = re.search(pat, full)
        if m:
            (char_start, char_end) = (m.start(1), m.end(1))
            break
    if char_start is None:
        matches = list(re.finditer(f'(?<!\\d){re.escape(target)}(?!\\d)', full))
        if not matches:
            return None
        m = matches[-1]
        (char_start, char_end) = (m.start(), m.end())
    pos = 0
    start_i = end_i = None
    for (i, tok) in enumerate(pieces):
        nxt = pos + len(tok)
        if start_i is None and nxt > char_start:
            start_i = i
        if start_i is not None and nxt >= char_end:
            end_i = i + 1
            break
        pos = nxt
    if start_i is None or end_i is None:
        return None
    return (start_i, end_i)

def estimator_b_selected_id_logprob(token_logprobs: Sequence[TokenLogprob], selected_number: int) -> Optional[float]:
    span = find_selected_number_span(token_logprobs, selected_number)
    if span is None:
        return None
    (start, end) = span
    return float(sum((token_logprobs[i].logprob for i in range(start, end))))

def logprob_to_unit_interval(lp: float) -> float:
    if lp > 0:
        lp = 0.0
    try:
        return float(math.exp(lp))
    except OverflowError:
        return 0.0

def softmax(xs: Sequence[float], temperature: float=1.0) -> list[float]:
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

def estimator_c_softmax_mass_on_selected(candidate_scores: Sequence[float], selected_index_0based: int, temperature: float=1.0) -> tuple[float, list[float]]:
    probs = softmax(candidate_scores, temperature=temperature)
    if not 0 <= selected_index_0based < len(probs):
        raise IndexError(f'selected_index_0based={selected_index_0based} not in [0, {len(probs)})')
    return (float(probs[selected_index_0based]), probs)

def _norm_tok(t: str) -> str:
    return t.strip().upper()

def first_token_option_logprobs(token_logprobs: Sequence[TokenLogprob], options: Sequence[str]) -> dict[str, float]:
    wanted = {_norm_tok(o): o for o in options}
    found: dict[str, float] = {}
    for tok in token_logprobs:
        cands: list[tuple[str, float]] = [(tok.token, tok.logprob)]
        cands.extend(list(tok.top_logprobs or ()))
        if not any((_norm_tok(t) for (t, _) in cands)):
            continue
        for (raw, lp) in cands:
            key = _norm_tok(raw)
            if key in wanted:
                canon = wanted[key]
                prev = found.get(canon)
                if prev is None or lp > prev:
                    found[canon] = float(lp)
        if found:
            break
    return found

def binary_choice_probability(token_logprobs: Sequence[TokenLogprob], positive: str, negative: str) -> Optional[tuple[float, float, float]]:
    found = first_token_option_logprobs(token_logprobs, [positive, negative])
    if positive not in found or negative not in found:
        return None
    lp_pos = found[positive]
    lp_neg = found[negative]
    m = max(lp_pos, lp_neg)
    e_pos = math.exp(lp_pos - m)
    e_neg = math.exp(lp_neg - m)
    z = e_pos + e_neg
    return (float(e_pos / z), float(lp_pos), float(lp_neg))
