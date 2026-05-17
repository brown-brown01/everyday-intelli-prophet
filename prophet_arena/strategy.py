"""Trade decision + sizing for the Prophet Arena bot.

Given a tempered belief P(YES) and a market quote, decide whether the
edge clears the EV threshold, pick the side, and size the position with
fractional Kelly — shrunk by belief uncertainty and capped by the
Prophet Arena ruleset limits.

Execution convention (Prophet Arena): BUY YES fills at ``best_ask``,
BUY NO fills at ``1 - best_bid``. Each share pays $1 if its side wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .bayes import Belief
from .config import Config


@dataclass
class Decision:
    market_id: str
    action: str    # always "BUY" — Prophet Arena positions are opened, not shorted
    side: str      # "YES" or "NO"
    shares: int
    price: float   # expected fill price
    edge: float    # expected value per share
    kelly: float   # full Kelly fraction, before fractional + uncertainty shrink
    p_yes: float
    rationale: str


@dataclass
class DrawdownState:
    """Current portfolio drawdown for Kelly shrinkage.

    ``current_dd`` is a fraction in [0, 1] of peak equity. ``dd_max``
    is the threshold at which sizing is fully halted; defaults match
    the live-bot risk knob ``cfg.dd_max``.
    """

    current_dd: float = 0.0
    dd_max: float = 0.25


def kelly_drawdown_adjustment(dd: float, dd_max: float = 0.25) -> float:
    """Linear Kelly shrinkage multiplier ``max(0, 1 - dd/dd_max)``.

    Returns 1.0 when at peak; 0.0 once drawdown hits ``dd_max``. Falls
    back to 1.0 when ``dd_max`` is non-positive (feature disabled).
    """
    if dd_max <= 0:
        return 1.0
    return max(0.0, 1.0 - float(dd) / float(dd_max))


def choose_risk_fraction(dd: float, f_min: float, f_max: float) -> float:
    """Adaptive risk-fraction in [f_min, f_max] based on drawdown.

    At peak (dd <= 0), run at ``f_max``. At deep drawdown (dd >= 10%),
    run at ``f_min``. Between those points, interpolate linearly.
    """
    if f_max <= f_min:
        return f_min
    if dd <= 0.0:
        return f_max
    if dd >= 0.10:
        return f_min
    # Interpolate from f_max -> f_min over dd in [0.0, 0.10].
    return f_max - (f_max - f_min) * (dd / 0.10)


def decide(
    market_id: str,
    belief: Belief,
    best_bid: float,
    best_ask: float,
    bankroll: float,
    cfg: Config,
    *,
    per_trade_cap: float,
    held_side: str | None = None,
    gross_room: float | None = None,
    dd_state: DrawdownState | None = None,
) -> Decision | None:
    """Return a trade Decision, or None to HOLD."""
    p = belief.p_yes
    best_bid = max(0.01, min(0.99, best_bid))
    best_ask = max(0.01, min(0.99, best_ask))

    # EV per share. BUY YES costs best_ask, pays $1 if YES.
    # BUY NO costs 1-best_bid, pays $1 if NO  →  EV = best_bid - p.
    ev_yes = p - best_ask
    ev_no = best_bid - p

    if ev_yes >= ev_no and ev_yes > cfg.ev_threshold:
        side, price, edge = "YES", best_ask, ev_yes
        kelly = (p - best_ask) / max(1e-6, 1.0 - best_ask)
    elif ev_no > cfg.ev_threshold:
        side, price, edge = "NO", 1.0 - best_bid, ev_no
        kelly = (best_bid - p) / max(1e-6, best_bid)
    else:
        return None

    # Cannot hold both sides of one market — the server rejects it.
    if held_side and held_side != side:
        return None

    kelly = max(0.0, min(1.0, kelly))
    # A noisy belief bets smaller: shrink by forecast uncertainty.
    shrink = max(0.0, 1.0 - 2.0 * belief.uncertainty)
    dd_mult = (
        kelly_drawdown_adjustment(dd_state.current_dd, dd_state.dd_max)
        if dd_state is not None
        else 1.0
    )
    frac = cfg.kelly_fraction * kelly * shrink * dd_mult

    notional = min(frac * bankroll, per_trade_cap)
    if gross_room is not None:
        notional = min(notional, max(0.0, gross_room))
    shares = math.floor(notional / max(price, 0.01))
    if shares < cfg.min_shares:
        return None

    dd_note = f" dd={dd_mult:.2f}" if dd_state is not None else ""
    rationale = (
        f"p_yes={p:.3f} prior={belief.prior:.3f} p_llm={belief.p_llm:.3f} "
        f"-> BUY {side} @ {price:.3f}  edge={edge:+.3f}  "
        f"kelly={kelly:.3f}x{shrink:.2f}{dd_note}  {shares} sh"
    )
    return Decision(market_id, "BUY", side, shares, price, edge, kelly, p, rationale)
