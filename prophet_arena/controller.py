"""Price-convergence adaptive controller for ``llm_weight``.

In the live Prophet Arena window almost no markets resolve (median time
to resolution ~1.6 years), so resolution outcomes give no usable signal.
The only signal available is *price movement* — each tick re-prices all
256 markets, and mark-to-market drift is exactly what the benchmark
scores you on.

This controller measures, in log-odds space, whether the **raw LLM
forecast** predicts the market's own subsequent drift, and slowly tunes
``llm_weight`` toward that predictive coefficient:

    s = logit(p_llm)      - logit(mid_t)        # raw LLM disagreement
    d = logit(mid_{t+k})  - logit(mid_t)        # realized k-tick drift
    beta = sum(s*d) / sum(s*s)                  # OLS through the origin

``beta`` is the fraction of our LLM-deviation the market realizes over
the markout horizon. ``llm_weight`` is moved toward ``clamp(beta)`` by a
slow EMA — a **move-toward-target** update, which converges to an
interior point (a level-driven nudge would just ratchet to a bound).
The update is gated on a block-bootstrap CI excluding zero and a minimum
sample, and frozen in the final stretch of the run.

Design choices, from the math review:
  * scores the RAW ``p_llm`` — the thing ``llm_weight`` controls — not
    the blended posterior (scoring the posterior would be circular);
  * works in log-odds, consistent with ``bayes.temper``;
  * move-toward-target ``alpha <- (1-eta)*alpha + eta*clamp(beta)``;
  * block-bootstrap by family for an autocorrelation-robust CI;
  * one non-overlapping markout per market — no inflated sample counts.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .bayesian_core import prob_to_log_odds

logger = logging.getLogger("prophet.controller")

# ── constants ──
TICK_MINUTES = 15
CONTROLLER_INTERVAL_TICKS = 96   # run a controller step ~once per day
MARKOUT_HORIZON_TICKS = 96       # measure drift one day after each forecast
FREEZE_TICKS = 384               # no updates in the final ~4 days of the run
MIN_OBS = 50                     # minimum matured markouts before tuning
ETA = 0.25                       # EMA rate: alpha closes 25% of the gap to beta
ALPHA_MIN, ALPHA_MAX = 0.40, 0.85
_BOOTSTRAP = 300


@dataclass
class ControllerState:
    """Persisted state of the ``llm_weight`` controller."""

    alpha: float                       # current live llm_weight
    tick_count: int = 0                # ticks processed (cumulative, survives restart)
    last_step_tick: int = 0            # tick_count at the last controller step
    history: list[dict] = field(default_factory=list)  # change log (judge-facing)


def load_state(path: Path, default_alpha: float) -> ControllerState:
    """Load controller state, or seed a fresh one at ``default_alpha``."""
    try:
        raw = json.loads(Path(path).read_text())
        return ControllerState(
            alpha=float(raw["alpha"]),
            tick_count=int(raw.get("tick_count", 0)),
            last_step_tick=int(raw.get("last_step_tick", 0)),
            history=list(raw.get("history", [])),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ControllerState(alpha=float(default_alpha))


def save_state(path: Path, state: ControllerState) -> None:
    """Atomically persist controller state (survives a bot restart)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps({
        "alpha": state.alpha,
        "tick_count": state.tick_count,
        "last_step_tick": state.last_step_tick,
        "history": state.history,
    }, indent=2))
    tmp.replace(path)


def log_forecasts(log_path: Path, tick_ts, records: list[dict]) -> None:
    """Append this tick's per-market ``(p_llm, mid)`` to the forecast log.

    ``records`` are the bot's per-market audit dicts; each needs
    ``market_id``, ``p_llm`` and ``prior`` (the market mid-price).
    ``family`` is used for autocorrelation-robust bootstrapping.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    ts = str(tick_ts)
    with open(log_path, "a") as f:
        for r in records:
            p_llm = r.get("p_llm")
            mid = r.get("prior")
            if p_llm is None or mid is None:
                continue
            f.write(json.dumps({
                "ts": ts,
                "market_id": r["market_id"],
                "family": r.get("family"),
                "p_llm": float(p_llm),
                "mid": float(mid),
            }) + "\n")


def _read_log(log_path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        text = Path(log_path).read_text()
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            d["ts"] = datetime.fromisoformat(str(d["ts"]).replace("Z", "+00:00"))
            out.append(d)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return out


def _ols_origin(obs: list[tuple[float, float, str]]) -> float:
    """OLS slope through the origin: ``beta = sum(s*d) / sum(s*s)``."""
    num = sum(s * d for s, d, _ in obs)
    den = sum(s * s for s, _, _ in obs)
    return num / den if den > 1e-12 else 0.0


def _bootstrap_ci(obs, n_boot=_BOOTSTRAP, lo=5, hi=95) -> tuple[float, float]:
    """Block-bootstrap CI for beta, resampling whole families."""
    blocks: dict[str, list] = defaultdict(list)
    for o in obs:
        blocks[o[2]].append(o)
    keys = list(blocks)
    if len(keys) < 2:
        return (0.0, 0.0)  # too few blocks to bootstrap -> CI includes 0 -> hold
    rng = random.Random(12345)  # fixed seed: a controller step is reproducible
    betas = []
    for _ in range(n_boot):
        sample: list = []
        for _ in range(len(keys)):
            sample.extend(blocks[rng.choice(keys)])
        betas.append(_ols_origin(sample))
    betas.sort()
    return (betas[int(lo / 100 * n_boot)], betas[int(hi / 100 * n_boot)])


def estimate_beta(log_path: Path, horizon_ticks: int = MARKOUT_HORIZON_TICKS) -> dict:
    """Estimate beta from the forecast log via non-overlapping markouts."""
    entries = _read_log(log_path)
    by_market: dict[str, list] = defaultdict(list)
    for e in entries:
        by_market[e["market_id"]].append(e)

    horizon = timedelta(minutes=TICK_MINUTES * horizon_ticks)
    obs: list[tuple[float, float, str]] = []
    for market_id, rows in by_market.items():
        rows.sort(key=lambda r: r["ts"])
        i = 0
        while i < len(rows):
            anchor = rows[i]
            j = i + 1
            while j < len(rows) and rows[j]["ts"] < anchor["ts"] + horizon:
                j += 1
            if j >= len(rows):
                break  # markout for this anchor has not matured yet
            match = rows[j]
            s = prob_to_log_odds(anchor["p_llm"]) - prob_to_log_odds(anchor["mid"])
            d = prob_to_log_odds(match["mid"]) - prob_to_log_odds(anchor["mid"])
            obs.append((s, d, str(anchor.get("family") or market_id)))
            i = j  # non-overlapping window: next anchor is this match

    if not obs:
        return {"beta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    beta = _ols_origin(obs)
    ci_low, ci_high = _bootstrap_ci(obs)
    return {"beta": beta, "ci_low": ci_low, "ci_high": ci_high, "n": len(obs)}


def maybe_step(state: ControllerState, log_path: Path, n_ticks: int) -> dict | None:
    """Run a controller step if one is due. Returns the change record, or None.

    A step is due every ``CONTROLLER_INTERVAL_TICKS``. It updates
    ``state.alpha`` only when the markout sample is large enough and the
    bootstrap CI for beta excludes zero, and never in the frozen final
    stretch. Every step (update or hold) is appended to ``state.history``.
    """
    if state.tick_count - state.last_step_tick < CONTROLLER_INTERVAL_TICKS:
        return None
    state.last_step_tick = state.tick_count

    frozen = n_ticks > FREEZE_TICKS and state.tick_count > (n_ticks - FREEZE_TICKS)
    est = estimate_beta(log_path)
    ci_excludes_zero = est["ci_low"] > 0.0 or est["ci_high"] < 0.0

    rec = {
        "tick": state.tick_count,
        "beta": round(est["beta"], 4),
        "ci": [round(est["ci_low"], 4), round(est["ci_high"], 4)],
        "n": est["n"],
        "alpha_before": round(state.alpha, 4),
    }
    if frozen:
        rec["action"] = "hold (frozen — final stretch)"
    elif est["n"] < MIN_OBS:
        rec["action"] = f"hold (n={est['n']} < {MIN_OBS})"
    elif not ci_excludes_zero:
        rec["action"] = "hold (CI includes 0)"
    else:
        target = min(max(est["beta"], ALPHA_MIN), ALPHA_MAX)
        state.alpha = (1.0 - ETA) * state.alpha + ETA * target
        rec["action"] = f"updated (target={target:.3f})"
    rec["alpha_after"] = round(state.alpha, 4)
    state.history.append(rec)
    logger.info(
        "controller @tick %d: beta=%.3f ci=[%.3f,%.3f] n=%d  alpha %.3f -> %.3f  [%s]",
        rec["tick"], est["beta"], est["ci_low"], est["ci_high"], est["n"],
        rec["alpha_before"], rec["alpha_after"], rec["action"],
    )
    return rec
