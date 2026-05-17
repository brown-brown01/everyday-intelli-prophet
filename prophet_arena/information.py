"""Information-theoretic helpers for the Prophet Arena forecaster.

Diagnostics over a single market's LLM forecast samples (Shannon entropy,
ensemble diversity, redundancy pruning) and a cross-market binned mutual
information estimator used by the harness to spot which sample slots are
redundant. Diversity drives :func:`prune_redundant`, which lets the
forecaster skip LLM samples that don't add signal — cheaper without
losing calibration.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from .bayes import Forecast


def _clip01(p: float) -> float:
    return float(min(max(p, 1e-9), 1.0 - 1e-9))


def entropy(p) -> float:
    """Shannon entropy in nats.

    Accepts a binary probability (scalar in [0, 1]) or a categorical
    distribution (iterable summing to ~1). For a uniform distribution
    over ``n`` outcomes the result is ``ln(n)``.
    """
    if isinstance(p, (int, float)):
        q = _clip01(float(p))
        return -(q * math.log(q) + (1 - q) * math.log(1 - q))
    arr = np.asarray(list(p), dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    arr = arr / arr.sum()
    return float(-np.sum(arr * np.log(arr)))


def binned_mutual_information(
    xs: Iterable[float], ys: Iterable[float], bins: int = 10
) -> float:
    """Estimate I(X; Y) for two probability series using equal-width bins.

    Returns ``H(X) + H(Y) - H(X, Y)`` in nats. Symmetric and ``>= 0``
    up to estimator noise. With fewer than 2 samples or a degenerate
    series returns 0.
    """
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    edges[-1] = 1.0 + 1e-9  # include p=1.0 in the last bin
    xi = np.digitize(x, edges) - 1
    yi = np.digitize(y, edges) - 1
    xi = np.clip(xi, 0, bins - 1)
    yi = np.clip(yi, 0, bins - 1)
    joint = np.zeros((bins, bins), dtype=float)
    for a, b in zip(xi, yi):
        joint[a, b] += 1.0
    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            pij = joint[i, j]
            if pij <= 0 or px[i] <= 0 or py[j] <= 0:
                continue
            mi += pij * math.log(pij / (px[i] * py[j]))
    return max(mi, 0.0)


def ensemble_diversity(forecasts: list[Forecast]) -> float:
    """Mean pairwise absolute distance across forecast probabilities.

    Identical samples → 0. Maximally spread samples (e.g. one at 0,
    one at 1) → close to 1.0. With ``<2`` forecasts returns 0.
    """
    if len(forecasts) < 2:
        return 0.0
    probs = np.array([_clip01(f.p_yes) for f in forecasts], dtype=float)
    n = len(probs)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(probs[i] - probs[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def prune_redundant(
    forecasts: list[Forecast], min_distance: float = 0.02
) -> list[Forecast]:
    """Drop near-duplicate forecasts (greedy farthest-keeps wins).

    The first sample is always kept. Subsequent samples are kept only
    if their probability is at least ``min_distance`` away from every
    already-kept sample. Empty/singleton inputs pass through unchanged.
    """
    if len(forecasts) <= 1 or min_distance <= 0.0:
        return list(forecasts)
    kept: list[Forecast] = [forecasts[0]]
    for f in forecasts[1:]:
        p = _clip01(f.p_yes)
        if all(abs(p - _clip01(k.p_yes)) >= min_distance for k in kept):
            kept.append(f)
    return kept


def cross_sample_redundancy(
    sample_matrix: list[list[float]], bins: int = 10
) -> list[list[float]]:
    """MI matrix across sample *slots*, computed over many markets.

    ``sample_matrix[m][k]`` is sample slot ``k`` for market ``m``.
    Returns an ``S × S`` matrix where ``S`` is the number of slots.
    Off-diagonal pairs with high MI indicate slots that move together
    across markets — i.e. self-consistency runs are reproducing each
    other rather than exploring. Diagonal is the per-slot entropy.
    """
    if not sample_matrix:
        return []
    n_slots = len(sample_matrix[0])
    columns = [[row[k] for row in sample_matrix if len(row) > k] for k in range(n_slots)]
    out: list[list[float]] = []
    for i in range(n_slots):
        row: list[float] = []
        for j in range(n_slots):
            if i == j:
                row.append(entropy(np.mean(columns[i])) if columns[i] else 0.0)
            else:
                row.append(binned_mutual_information(columns[i], columns[j], bins=bins))
        out.append(row)
    return out
