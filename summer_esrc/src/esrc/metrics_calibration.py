from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

@dataclass(frozen=True)
class CalibrationMetrics:
    n: int
    n_correct: int
    top1_accuracy: float
    ece: float
    brier: float
    average_precision: float
    recall_at_90_precision: Optional[float]
    recall_at_99_precision: Optional[float]

@dataclass(frozen=True)
class ReliabilityBin:
    bin_lo: float
    bin_hi: float
    n: int
    mean_confidence: float
    accuracy: float
    gap: float

@dataclass(frozen=True)
class McNemarResult:
    n11: int
    n10: int
    n01: int
    n00: int
    statistic: float
    p_value: float
    n_discordant: int
    note: str

@dataclass(frozen=True)
class BootstrapCI:
    point: Optional[float]
    mean: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_undefined: int
    undefined_as: str

@dataclass(frozen=True)
class ThresholdEval:
    threshold: float
    n_accepted: int
    n_correct_accepted: int
    precision: Optional[float]
    recall: Optional[float]

def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int=10) -> float:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float('nan')
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        (lo, hi) = (bins[i], bins[i + 1])
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += mask.sum() / n * abs(acc - avg_conf)
    return float(ece)

def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float('nan')
    return float(np.mean((conf - y) ** 2))

def recall_at_precision(confidences: Sequence[float], correct: Sequence[bool], target_precision: float) -> Optional[float]:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=int)
    if y.sum() == 0:
        return None
    (precision, recall, _thresholds) = precision_recall_curve(y, conf)
    ok = precision[:-1] >= target_precision
    if not np.any(ok):
        if float(precision[-1]) >= target_precision:
            return float(recall[-1])
        return None
    return float(np.max(recall[:-1][ok]))

def evaluate_scores(confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int=10) -> CalibrationMetrics:
    conf = [float(c) for c in confidences]
    y = [bool(v) for v in correct]
    n = len(conf)
    n_correct = sum(y)
    ap = float('nan')
    if n and sum(y) > 0 and (sum(y) < n):
        ap = float(average_precision_score(y, conf))
    elif n and sum(y) == n:
        ap = 1.0
    return CalibrationMetrics(n=n, n_correct=n_correct, top1_accuracy=n_correct / n if n else float('nan'), ece=expected_calibration_error(conf, y, n_bins=n_bins), brier=brier_score(conf, y), average_precision=ap, recall_at_90_precision=recall_at_precision(conf, y, 0.9), recall_at_99_precision=recall_at_precision(conf, y, 0.99))

def reliability_table(confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int=10) -> list[ReliabilityBin]:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        (lo, hi) = (float(edges[i]), float(edges[i + 1]))
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            out.append(ReliabilityBin(lo, hi, 0, float('nan'), float('nan'), float('nan')))
            continue
        mean_c = float(conf[mask].mean())
        acc = float(y[mask].mean())
        out.append(ReliabilityBin(lo, hi, n, mean_c, acc, abs(acc - mean_c)))
    return out

def mcnemar_threshold_match(conf_a: Sequence[float], conf_b: Sequence[float], correct: Sequence[bool], *, threshold: float=0.9, continuity_correction: bool=True) -> McNemarResult:
    a = np.asarray(conf_a, dtype=float)
    b = np.asarray(conf_b, dtype=float)
    y = np.asarray(correct, dtype=bool)
    if not len(a) == len(b) == len(y):
        raise ValueError('conf_a, conf_b, correct length mismatch')
    pred_a = a >= threshold
    pred_b = b >= threshold
    match_a = pred_a == y
    match_b = pred_b == y
    n11 = int(np.sum(match_a & match_b))
    n10 = int(np.sum(match_a & ~match_b))
    n01 = int(np.sum(~match_a & match_b))
    n00 = int(np.sum(~match_a & ~match_b))
    disc = n10 + n01
    if disc == 0:
        return McNemarResult(n11=n11, n10=n10, n01=n01, n00=n00, statistic=0.0, p_value=1.0, n_discordant=0, note=f'No discordant pairs at τ={threshold}; McNemar undefined / trivially equal.')
    diff = abs(n10 - n01)
    if continuity_correction:
        diff = max(diff - 1, 0)
    stat = diff ** 2 / disc
    from math import erfc, sqrt
    p = float(erfc(sqrt(stat / 2.0))) if stat > 0 else 1.0
    return McNemarResult(n11=n11, n10=n10, n01=n01, n00=n00, statistic=float(stat), p_value=p, n_discordant=disc, note=f'Thresholded classifiers at τ={threshold}: predict correct iff conf≥τ. Continuity correction={continuity_correction}.')

def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    if n_tests < 1:
        raise ValueError('n_tests must be >= 1')
    return alpha / n_tests

def bootstrap_metric_ci(confidences: Sequence[float], correct: Sequence[bool], metric_fn, *, n_boot: int=10000, seed: int=2026, undefined_as_zero: bool=True) -> BootstrapCI:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=int)
    rng = np.random.default_rng(seed)
    point = metric_fn(conf, y)
    n = len(y)
    vals: list[float] = []
    n_undef = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = metric_fn(conf[idx], y[idx])
        if v is None or (isinstance(v, float) and np.isnan(v)):
            n_undef += 1
            if undefined_as_zero:
                vals.append(0.0)
            continue
        vals.append(float(v))
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return BootstrapCI(point=None if point is None else float(point), mean=float('nan'), ci_low=float('nan'), ci_high=float('nan'), n_boot=n_boot, n_undefined=n_undef, undefined_as='omit')
    return BootstrapCI(point=None if point is None else float(point), mean=float(arr.mean()), ci_low=float(np.quantile(arr, 0.025)), ci_high=float(np.quantile(arr, 0.975)), n_boot=n_boot, n_undefined=n_undef, undefined_as='zero' if undefined_as_zero else 'omit')

def evaluate_at_threshold(confidences: Sequence[float], correct: Sequence[bool], threshold: float) -> ThresholdEval:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=bool)
    accept = conf >= threshold
    n_acc = int(accept.sum())
    n_tp = int((accept & y).sum())
    n_pos = int(y.sum())
    precision = n_tp / n_acc if n_acc else None
    recall = n_tp / n_pos if n_pos else None
    return ThresholdEval(threshold=float(threshold), n_accepted=n_acc, n_correct_accepted=n_tp, precision=precision, recall=recall)

def fit_isotonic(confidences: Sequence[float], correct: Sequence[bool]) -> IsotonicRegression:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(conf, y)
    return iso

def calibrate_scores(iso: IsotonicRegression, confidences: Sequence[float]) -> np.ndarray:
    return np.asarray(iso.predict(np.asarray(confidences, dtype=float)), dtype=float)

def threshold_for_precision(confidences: Sequence[float], correct: Sequence[bool], target_precision: float) -> Optional[float]:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=int)
    if y.sum() == 0:
        return None
    (precision, _recall, thresholds) = precision_recall_curve(y, conf)
    ok = precision[:-1] >= target_precision
    if not np.any(ok):
        return None
    recall = _recall[:-1]
    best_i = int(np.argmax(np.where(ok, recall, -1.0)))
    return float(thresholds[best_i])
