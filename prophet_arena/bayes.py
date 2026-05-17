"""Bayesian tempering layer for the Prophet Arena forecaster.

Combines the prior (market mid-price when live, base rate when scoring
offline) with one or more LLM probability estimates as a logarithmic
opinion pool in log-odds space — the same numerically stable
representation the V1 belief engine uses.

Built directly on :class:`prophet_arena.bayesian_core.BayesianUpdater`
so the hackathon bot and the V1 analytical stack share one Bayesian core.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bayesian_core import (
    BayesianUpdater,
    UpdateRecord,
    log_odds_to_prob,
    prob_to_log_odds,
)

P_MIN = BayesianUpdater.P_MIN
P_MAX = BayesianUpdater.P_MAX


def _clip(p: float) -> float:
    return float(min(max(p, P_MIN), P_MAX))


@dataclass
class Forecast:
    """One forecaster's estimate of P(YES)."""

    p_yes: float
    source: str = "llm"
    rationale: str = ""


@dataclass
class Belief:
    """A tempered posterior plus the pieces that produced it."""

    p_yes: float        # posterior after Bayesian tempering
    prior: float        # market mid (live) or base rate (offline)
    p_llm: float        # mean raw LLM forecast, pre-temper
    uncertainty: float  # forecast spread — shrinks Kelly sizing
    n_forecasts: int
    forecasts: list[Forecast] = field(default_factory=list)


class ForecastPool(BayesianUpdater):
    """Log-odds opinion pool built on the V1 :class:`BayesianUpdater`.

    Seeded with a prior; :meth:`fold` moves the belief a ``weight``
    fraction of the way toward a forecast in log-odds space. ``weight=1``
    adopts the forecast outright; ``weight<1`` keeps it anchored to the
    prior — Bayesian tempering of an overconfident model.
    """

    def _apply_log_odds(self, lo: float, source: str, weight: float) -> None:
        prior = self.posterior
        # log P(YES) and log P(NO) derived from a log-odds value, normalized
        self.log_prior_yes = -float(np.logaddexp(0.0, -lo))
        self.log_prior_no = -float(np.logaddexp(0.0, lo))
        self.num_updates += 1
        self.update_history.append(
            UpdateRecord(source, weight, prior, self.posterior, self.num_updates)
        )

    def fold(self, p_yes: float, weight: float = 1.0, source: str = "llm") -> float:
        """Move the belief a ``weight`` fraction toward ``p_yes``."""
        weight = float(min(max(weight, 0.0), 1.0))
        target = prob_to_log_odds(_clip(p_yes))
        blended = (1.0 - weight) * self.log_odds + weight * target
        self._apply_log_odds(blended, source, weight)
        return self.posterior


def geometric_mean_pool(probs: list[float]) -> float:
    """Geometric mean of probabilities: ``(∏ pᵢ)^(1/n)``.

    Equivalent to averaging in log space. Returned value is clipped to
    ``[P_MIN, P_MAX]``. Empty list returns 0.5.
    """
    if not probs:
        return 0.5
    clipped = [_clip(p) for p in probs]
    log_mean = float(np.mean([math.log(p) for p in clipped]))
    return _clip(math.exp(log_mean))


def log_odds_pool(
    probs: list[float], weights: list[float] | None = None
) -> float:
    """Logarithmic opinion pool of probabilities.

    Computes ``softmax( Σ wᵢ·logit(pᵢ) / Σ wᵢ )``. Equal weights when
    ``weights is None`` — that is the unweighted log-odds mean and the
    log-linear opinion pool used by Bayesian tempering. Returned value
    is clipped to ``[P_MIN, P_MAX]``. Empty list returns 0.5.
    """
    if not probs:
        return 0.5
    logits = np.array([prob_to_log_odds(_clip(p)) for p in probs], dtype=float)
    if weights is None:
        mean_logit = float(np.mean(logits))
    else:
        w = np.array(weights, dtype=float)
        if w.shape != logits.shape:
            raise ValueError(
                f"weights length {w.shape} != probs length {logits.shape}"
            )
        total = float(np.sum(w))
        if total <= 0:
            mean_logit = float(np.mean(logits))
        else:
            mean_logit = float(np.sum(w * logits) / total)
    return _clip(log_odds_to_prob(mean_logit))


def temper(
    prior: float,
    forecasts: list[Forecast],
    llm_weight: float = 0.7,
    llm_temperature: float = 1.0,
    *,
    sample_weights: list[float] | None = None,
    regime_entropy: float = 0.0,
    regime_weight: float = 0.5,
) -> Belief:
    """Temper LLM forecasts with a Bayesian prior.

    The forecasts are averaged in log-odds space (optionally with
    ``sample_weights``), optionally flattened by ``llm_temperature``
    (>1 treats the model as overconfident), then folded into a
    :class:`ForecastPool` seeded with ``prior`` at ``llm_weight``.
    With ``llm_weight=1`` the posterior equals the LLM forecast;
    lower values pull it back toward the prior.

    ``regime_entropy`` (default 0.0) folds latent-regime uncertainty
    into ``Belief.uncertainty`` via ``sqrt(spread² + w·entropy²)``,
    which then shrinks Kelly sizing downstream.
    """
    prior = _clip(prior)
    if not forecasts:
        return Belief(prior, prior, prior, 0.20, 0, [])

    probs = [f.p_yes for f in forecasts]
    p_llm = log_odds_pool(probs, weights=sample_weights)
    mean_logit = prob_to_log_odds(p_llm)
    cooked = log_odds_to_prob(mean_logit / max(llm_temperature, 1e-6))

    pool = ForecastPool(prior=prior)
    posterior = pool.fold(cooked, weight=llm_weight, source="llm")

    spread = (
        float(np.std([_clip(f.p_yes) for f in forecasts]))
        if len(forecasts) > 1
        else 0.0
    )
    combined = math.sqrt(spread ** 2 + max(regime_weight, 0.0) * (regime_entropy ** 2))
    uncertainty = max(combined, 0.06)
    return Belief(posterior, prior, p_llm, uncertainty, len(forecasts), list(forecasts))
