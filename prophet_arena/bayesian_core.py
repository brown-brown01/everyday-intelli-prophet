"""Sequential Bayesian updater for prediction market beliefs.

Maintains beliefs in log-odds space for numerical stability during
sequential updates. Supports multiple evidence types with configurable
likelihood weights.

Key reference: Sequential Bayesian updating (Oravecz 2016)
P(H|D_1,...,D_t) ∝ P(H) * Π P(D_k|H)
In log-space: log P(H|D) = log P(H) + Σ log P(D_k|H) - log Z
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class UpdateRecord:
    """Record of a single Bayesian update for auditability."""

    event_type: str
    confidence: float
    prior: float
    posterior: float
    update_n: int


class BayesianUpdater:
    """Sequential Bayesian updater for binary prediction market beliefs.

    Works in log-odds space internally to prevent underflow/overflow.
    log_odds = ln(p / (1 - p))
    p = 1 / (1 + exp(-log_odds))
    """

    # Default likelihood weights per event type.
    # P(evidence | YES) and P(evidence | NO).
    # These are prior estimates — autoresearch will calibrate them.
    DEFAULT_LIKELIHOODS: dict[str, dict[str, float]] = {
        "military_deployment": {"YES": 0.75, "NO": 0.25},
        "troop_movement": {"YES": 0.70, "NO": 0.30},
        "sanctions_announcement": {"YES": 0.60, "NO": 0.45},
        "diplomatic_statement": {"YES": 0.40, "NO": 0.55},
        "intel_leak": {"YES": 0.65, "NO": 0.35},
        "ceasefire": {"YES": 0.25, "NO": 0.75},
        "de_escalation": {"YES": 0.20, "NO": 0.80},
        # Pharos-specific event types
        "MILITARY": {"YES": 0.75, "NO": 0.25},
        "DIPLOMATIC": {"YES": 0.40, "NO": 0.55},
        "INTELLIGENCE": {"YES": 0.65, "NO": 0.35},
        "ECONOMIC": {"YES": 0.55, "NO": 0.45},
        "HUMANITARIAN": {"YES": 0.45, "NO": 0.50},
        "POLITICAL": {"YES": 0.50, "NO": 0.50},
    }

    # Probability clamp bounds to avoid log(0)
    P_MIN = 0.001
    P_MAX = 0.999

    def __init__(
        self,
        prior: float = 0.5,
        likelihoods: dict[str, dict[str, float]] | None = None,
    ):
        """Initialize with a prior probability.

        Args:
            prior: Initial P(YES), between 0 and 1.
            likelihoods: Custom likelihood weights per event type.
                         Falls back to DEFAULT_LIKELIHOODS if not provided.
        """
        prior = np.clip(prior, self.P_MIN, self.P_MAX)
        self.log_prior_yes = float(np.log(prior))
        self.log_prior_no = float(np.log(1 - prior))
        self.likelihoods = likelihoods or self.DEFAULT_LIKELIHOODS.copy()
        self.num_updates = 0
        self.update_history: list[UpdateRecord] = []

    def update(self, event_type: str, confidence: float = 1.0) -> float:
        """Update belief given a new evidence event.

        Args:
            event_type: Type of event (must match a key in likelihoods).
            confidence: Scaling factor for the update (0 to 1).
                       Lower confidence = smaller update.

        Returns:
            New posterior P(YES) after the update.
        """
        if event_type not in self.likelihoods:
            return self.posterior  # Unknown event type — no update

        prior = self.posterior
        lh = self.likelihoods[event_type]

        # Scale likelihood by confidence: when confidence < 1,
        # the evidence is weaker (likelihood ratio closer to 1)
        log_lh_yes = confidence * np.log(max(lh["YES"], 1e-10))
        log_lh_no = confidence * np.log(max(lh["NO"], 1e-10))

        self.log_prior_yes += log_lh_yes
        self.log_prior_no += log_lh_no

        # Normalize via log-sum-exp
        log_z = float(np.logaddexp(self.log_prior_yes, self.log_prior_no))
        self.log_prior_yes -= log_z
        self.log_prior_no -= log_z

        self.num_updates += 1
        self.update_history.append(
            UpdateRecord(
                event_type=event_type,
                confidence=confidence,
                prior=prior,
                posterior=self.posterior,
                update_n=self.num_updates,
            )
        )
        return self.posterior

    def batch_update(
        self, events: list[tuple[str, float]]
    ) -> float:
        """Apply multiple updates in sequence.

        Args:
            events: List of (event_type, confidence) tuples.

        Returns:
            Final posterior after all updates.
        """
        for event_type, confidence in events:
            self.update(event_type, confidence)
        return self.posterior

    @property
    def posterior(self) -> float:
        """Current P(YES) belief."""
        p = float(np.exp(self.log_prior_yes))
        return float(np.clip(p, self.P_MIN, self.P_MAX))

    @property
    def log_odds(self) -> float:
        """Current log-odds: ln(P(YES) / P(NO))."""
        return self.log_prior_yes - self.log_prior_no

    def set_likelihoods(
        self,
        likelihoods: dict[str, dict[str, float]],
        merge: bool = True,
    ) -> None:
        """Override or extend per-event-type likelihoods.

        Args:
            likelihoods: Mapping ``event_type -> {"YES": p, "NO": p}``.
            merge: When True (default), the new likelihoods are merged into
                the existing dict so unspecified types keep their priors.
                When False, the dict is fully replaced.
        """
        if merge:
            self.likelihoods = {**self.likelihoods, **likelihoods}
        else:
            self.likelihoods = dict(likelihoods)

    def reset(self, prior: float = 0.5) -> None:
        """Reset to a new prior, clearing update history."""
        prior = np.clip(prior, self.P_MIN, self.P_MAX)
        self.log_prior_yes = float(np.log(prior))
        self.log_prior_no = float(np.log(1 - prior))
        self.num_updates = 0
        self.update_history.clear()

    def get_posterior_distribution(self) -> tuple[float, float]:
        """Return approximate mean and std of posterior for Kelly shrinkage.

        Uses update history to estimate uncertainty. More updates with
        consistent direction = lower variance.

        Returns:
            Tuple of (mean, std) of the posterior estimate.
        """
        if self.num_updates < 2:
            # High uncertainty with few updates
            return (self.posterior, 0.15)

        posteriors = [r.posterior for r in self.update_history]
        mean = float(np.mean(posteriors))
        std = float(np.std(posteriors))
        # Floor std at 0.02 to prevent overconfidence
        return (mean, max(std, 0.02))


def prob_to_log_odds(p: float) -> float:
    """Convert probability to log-odds."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(np.log(p / (1 - p)))


def log_odds_to_prob(lo: float) -> float:
    """Convert log-odds to probability."""
    return float(1.0 / (1.0 + np.exp(-lo)))
