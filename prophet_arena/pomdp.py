"""POMDP belief over latent market regimes (Prophet Arena).

Four discrete regimes — ``bull``, ``bear``, ``sideways``, ``volatile`` —
updated by Bayes from per-tick price-change observations:

    b'(s') = η · O(o|s') · Σₛ T(s'|s) · b(s)

Action space is ``{observe}`` only — value iteration is Tier 3 and out
of scope. Entropy of the belief feeds into :func:`bayes.temper` as an
extra uncertainty term so volatile regimes shrink Kelly sizing without
biasing the LLM-vs-prior weighting.

State persists per ``market_id`` in ``data/prophet/regime_state.json``;
the bot loads it at startup and saves after each tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

STATES: tuple[str, ...] = ("bull", "bear", "sideways", "volatile")
OBSERVATIONS: tuple[str, ...] = ("down_big", "down_small", "flat", "up_small", "up_big")

# Transition matrix T[from_state, to_state]. Regimes are sticky (~0.7 self),
# volatile bleeds back into the others, sideways is the absorbing "default".
_TRANSITION = np.array(
    [
        # to:  bull  bear  side  vol
        [0.70, 0.05, 0.15, 0.10],  # from bull
        [0.05, 0.70, 0.15, 0.10],  # from bear
        [0.10, 0.10, 0.70, 0.10],  # from sideways
        [0.20, 0.20, 0.20, 0.40],  # from volatile
    ],
    dtype=float,
)

# Observation matrix O[state, observation]. Bull skews up, bear skews down,
# sideways concentrates on flat, volatile spreads to the extremes.
_OBSERVATION = np.array(
    [
        # obs: dn_b  dn_s  flat  up_s  up_b
        [0.04, 0.10, 0.20, 0.36, 0.30],  # bull
        [0.30, 0.36, 0.20, 0.10, 0.04],  # bear
        [0.05, 0.15, 0.60, 0.15, 0.05],  # sideways
        [0.30, 0.10, 0.20, 0.10, 0.30],  # volatile
    ],
    dtype=float,
)

# Default observation bin edges (price-change fraction). Bin index from
# np.digitize maps directly to OBSERVATIONS via clamp-to-len.
DEFAULT_BINS: tuple[float, ...] = (-0.05, -0.01, 0.01, 0.05)


def discretize_change(delta: float, bins: tuple[float, ...] = DEFAULT_BINS) -> str:
    """Map a price-change fraction to a discrete observation label."""
    idx = int(np.digitize([float(delta)], np.asarray(bins, dtype=float))[0])
    idx = max(0, min(idx, len(OBSERVATIONS) - 1))
    return OBSERVATIONS[idx]


@dataclass
class RegimePOMDP:
    """Belief tracker over latent market regimes.

    ``belief`` is a length-4 probability vector over :data:`STATES`,
    always summing to 1. ``last_prior`` is the prior (e.g. market mid)
    observed on the most recent tick — the bot uses it to compute the
    next tick's price-change observation. ``last_seen`` is an ISO8601
    UTC timestamp the bot uses to evict stale markets.
    """

    belief: np.ndarray = field(
        default_factory=lambda: np.full(len(STATES), 1.0 / len(STATES))
    )
    n_updates: int = 0
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_prior: float | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.belief, dtype=float)
        if arr.shape != (len(STATES),):
            raise ValueError(f"belief must have shape ({len(STATES)},); got {arr.shape}")
        total = float(arr.sum())
        self.belief = arr / total if total > 0 else np.full(len(STATES), 1.0 / len(STATES))

    def update(self, observation: str) -> np.ndarray:
        """Apply one Bayesian belief update for a discrete observation.

        Returns the new belief vector (also stored on self).
        """
        if observation not in OBSERVATIONS:
            return self.belief
        o_idx = OBSERVATIONS.index(observation)
        predicted = _TRANSITION.T @ self.belief         # Σₛ T(s'|s)·b(s)
        likelihood = _OBSERVATION[:, o_idx]              # O(o|s') per s'
        unnormalized = likelihood * predicted
        z = float(unnormalized.sum())
        if z <= 0:
            return self.belief  # observation impossible under all states
        self.belief = unnormalized / z
        self.n_updates += 1
        self.last_seen = datetime.now(timezone.utc).isoformat()
        return self.belief

    def update_from_change(
        self, delta: float, bins: tuple[float, ...] = DEFAULT_BINS
    ) -> np.ndarray:
        """Update from a raw price-change fraction (binned to an observation)."""
        return self.update(discretize_change(delta, bins))

    def entropy(self) -> float:
        """Shannon entropy of the regime belief in nats. ``ln(|S|)`` for uniform."""
        b = np.clip(self.belief, 1e-12, 1.0)
        return float(-np.sum(b * np.log(b)))

    def most_likely(self) -> str:
        return STATES[int(np.argmax(self.belief))]

    def to_dict(self) -> dict:
        return {
            "belief": [float(x) for x in self.belief],
            "n_updates": int(self.n_updates),
            "last_seen": self.last_seen,
            "last_prior": None if self.last_prior is None else float(self.last_prior),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimePOMDP":
        lp = d.get("last_prior")
        return cls(
            belief=np.asarray(d.get("belief", [0.25] * len(STATES)), dtype=float),
            n_updates=int(d.get("n_updates", 0)),
            last_seen=str(d.get("last_seen", datetime.now(timezone.utc).isoformat())),
            last_prior=None if lp is None else float(lp),
        )


def load_states(path: Path) -> dict[str, RegimePOMDP]:
    """Load a ``{market_id: RegimePOMDP}`` map from disk, empty if missing."""
    try:
        raw = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {mid: RegimePOMDP.from_dict(d) for mid, d in raw.items()}


def save_states(path: Path, states: dict[str, RegimePOMDP]) -> None:
    """Atomically write a regime-state map to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {mid: pomdp.to_dict() for mid, pomdp in states.items()}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)


def evict_stale(states: dict[str, RegimePOMDP], max_age_hours: float = 24.0) -> int:
    """Drop entries whose ``last_seen`` is older than ``max_age_hours``.

    Returns the number of entries removed.
    """
    if max_age_hours <= 0:
        return 0
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - max_age_hours * 3600.0
    stale: list[str] = []
    for mid, pomdp in states.items():
        try:
            ts = datetime.fromisoformat(pomdp.last_seen).timestamp()
        except (ValueError, TypeError):
            ts = 0.0
        if ts < cutoff:
            stale.append(mid)
    for mid in stale:
        del states[mid]
    return len(stale)
