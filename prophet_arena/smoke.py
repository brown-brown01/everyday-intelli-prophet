"""End-to-end smoke test for the Prophet Arena bot.

Exercises the full decision pipeline on a LIVE market snapshot —
forecaster → families (coherence) → Bayesian temper → strategy → edge
ranking → risk caps → the adaptive controller — without creating an
experiment or submitting trades. (The submit/finalize/complete lifecycle
is verified separately against the live API.)

    python -m prophet_arena.smoke [n_markets]      # default 30; pass 256 to stress

Exits 0 on success, 1 on any failure.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ai_prophet_core import ServerAPIClient, TradeIntentRequest, ruleset

from . import controller
from .bayes import temper
from .config import load_config
from .families import JointForecaster, check_coherence
from .strategy import decide


def _fail(msg: str) -> None:
    print(f"  FAIL — {msg}")
    sys.exit(1)


def run(n_markets: int = 30) -> None:
    print("Prophet Arena bot — end-to-end smoke test\n")
    cfg = load_config()
    if not cfg.pa_api_key:
        _fail("PA_SERVER_API_KEY not set")

    # ── 1. live snapshot (read-only) ──
    api = ServerAPIClient(base_url=cfg.pa_server_url, api_key=cfg.pa_api_key, timeout=30)
    try:
        snap = api.get_market_snapshot()
    finally:
        api.close()
    markets = list(snap.markets)[:n_markets]
    if not markets:
        _fail("snapshot returned no markets")
    print(f"  [1] snapshot OK — {len(markets)} of {snap.market_count} markets")

    # ── 2. forecaster + family coherence ──
    forecasts = JointForecaster(cfg).forecast_markets(markets)
    missing = [m.market_id for m in markets if not forecasts.get(m.market_id)]
    if missing:
        _fail(f"{len(missing)} markets got no forecast")
    violations = check_coherence(forecasts)
    if violations:
        _fail(f"coherence violations: {violations[:3]}")
    print(f"  [2] forecaster OK — {len(forecasts)} forecasts, all probabilities in (0,1)")

    # ── 3. temper + decide + edge ranking ──
    ctrl = controller.ControllerState(alpha=cfg.llm_weight)
    per_trade_cap = cfg.starting_cash * cfg.per_trade_risk_cap
    decisions = []
    for m in markets:
        bid, ask = float(m.quote.best_bid), float(m.quote.best_ask)
        prior = max(0.01, min(0.99, (bid + ask) / 2.0))
        belief = temper(prior, forecasts[m.market_id], ctrl.alpha, cfg.llm_temperature)
        if not (0.0 < belief.p_yes < 1.0):
            _fail(f"{m.market_id}: tempered p_yes={belief.p_yes} out of range")
        d = decide(
            m.market_id, belief, bid, ask, cfg.starting_cash, cfg,
            per_trade_cap=per_trade_cap,
        )
        if d is not None:
            decisions.append(d)
    decisions.sort(key=lambda d: d.edge, reverse=True)
    cap = min(cfg.max_trades_per_tick, ruleset.MAX_TRADES_PER_TICK)
    top = decisions[:cap]
    print(f"  [3] decision pipeline OK — {len(decisions)} tradable, top {len(top)} by edge")

    # ── 4. risk caps on every intent ──
    intents = []
    for d in top:
        notional = d.shares * d.price
        if notional > per_trade_cap + 1e-6:
            _fail(f"{d.market_id}: notional ${notional:.0f} exceeds per-market cap")
        if d.shares < cfg.min_shares:
            _fail(f"{d.market_id}: {d.shares} shares below minimum")
        intents.append(TradeIntentRequest(
            market_id=d.market_id, action=d.action, side=d.side,
            shares=str(d.shares), idempotency_key="",
        ))
    print(f"  [4] risk caps OK — {len(intents)} intents, all within per-trade risk cap")

    # ── 5. adaptive controller (isolated temp state) ──
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "forecast_log.jsonl"
        audit = [{
            "market_id": m.market_id,
            "family": getattr(m, "family", None),
            "p_llm": forecasts[m.market_id][0].p_yes,
            "prior": max(0.01, min(0.99, (float(m.quote.best_bid)
                                          + float(m.quote.best_ask)) / 2.0)),
        } for m in markets]
        controller.log_forecasts(log, "2026-05-18T00:00:00+00:00", audit)
        est = controller.estimate_beta(log)
        st = controller.ControllerState(alpha=cfg.llm_weight, tick_count=96)
        controller.maybe_step(st, log, n_ticks=1344)
    print(f"  [5] controller OK — logged {len(audit)} forecasts, "
          f"estimate_beta ran (n={est['n']}), controller step ran")

    print(f"\nSMOKE PASSED — full pipeline runs end-to-end "
          f"(backend={cfg.resolved_backend()}, model={cfg.llm_model})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(n)
