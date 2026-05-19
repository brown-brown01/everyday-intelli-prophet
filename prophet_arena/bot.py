"""Prophet Arena tick-loop bot.

Runs the BenchmarkSession lifecycle: claim a 15-minute tick, forecast
every candidate market with the Bayesian-tempered LLM brain, size
trades with fractional Kelly, submit, finalize, advance. Resilient to
network blips; resumable — re-run with the same ``PA_SLUG`` and the
server returns the existing experiment.

    python -m prophet_arena.bot
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ai_prophet_core import ServerAPIClient, TradeIntentRequest, ruleset
from ai_prophet_core.arena import BenchmarkSession

from . import controller
from .bayes import temper
from .config import DATA_DIR, Config, load_config
from .families import JointForecaster
from .pomdp import RegimePOMDP, evict_stale, load_states, save_states
from .strategy import DrawdownState, choose_risk_fraction, decide, decide_exit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prophet.bot")

_REGIME_STATE_PATH = DATA_DIR / "regime_state.json"
_EQUITY_PEAK_PATH = DATA_DIR / "equity_peak.json"
_CONTROLLER_STATE_PATH = DATA_DIR / "controller_state.json"
_FORECAST_LOG_PATH = DATA_DIR / "forecast_log.jsonl"


def _load_equity_peak(slug: str) -> float:
    try:
        raw = json.loads(_EQUITY_PEAK_PATH.read_text())
        return float(raw.get(slug, 0.0))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def _save_equity_peak(slug: str, peak: float) -> None:
    try:
        raw = json.loads(_EQUITY_PEAK_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    raw[slug] = float(peak)
    _EQUITY_PEAK_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _EQUITY_PEAK_PATH.with_suffix(_EQUITY_PEAK_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(raw, indent=2))
    tmp.replace(_EQUITY_PEAK_PATH)


def _run_tick(
    session: BenchmarkSession,
    cfg: Config,
    forecaster,
    lease,
    idx: int,
    regime_states: dict[str, RegimePOMDP],
    equity_peak: dict[str, float],
    ctrl: controller.ControllerState,
) -> None:
    tick = session.load_candidates(lease)
    lease = tick.lease
    markets = tick.candidates.markets
    logger.info("Tick %s — %d candidate markets", lease.tick_id, len(markets))

    # Family-aware joint forecasting: one LLM call per mutual-exclusivity
    # group, normalized so sibling markets stay coherent (probabilities
    # sum to ~1) instead of independently over-pricing every longshot.
    forecasts = forecaster.forecast_markets(markets)

    portfolio = session.get_portfolio(idx)
    bankroll = float(portfolio.cash) if portfolio else cfg.starting_cash
    held: dict[str, str] = {}
    gross = 0.0
    equity = bankroll
    if portfolio:
        equity = float(getattr(portfolio, "equity", bankroll) or bankroll)
        for pos in portfolio.positions:
            held[pos.market_id] = pos.side
            gross += abs(float(pos.shares) * float(pos.avg_entry_price))

    # Drawdown vs. peak equity (unrealized + realized — bleeding books shrink Kelly).
    slug = cfg.slug
    prev_peak = equity_peak.get(slug, 0.0)
    peak = max(prev_peak, equity, cfg.starting_cash)
    equity_peak[slug] = peak
    dd = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
    dd_state = DrawdownState(current_dd=dd, dd_max=cfg.dd_max)
    if dd > 0:
        logger.info("Drawdown %.2f%% (peak $%.0f, equity $%.0f)", dd * 100, peak, equity)
    risk_f = choose_risk_fraction(dd, cfg.risk_fraction_min, cfg.risk_fraction_max)
    budget_from_equity = equity * risk_f
    max_api_exposure = ruleset.MAX_GROSS_EXPOSURE * cfg.api_safety_margin
    remaining_api_room = max_api_exposure - gross
    exposure_budget = max(0.0, min(budget_from_equity, remaining_api_room))
    per_trade_cap = equity * cfg.per_trade_risk_cap
    trade_cap = min(cfg.max_trades_per_tick, ruleset.MAX_TRADES_PER_TICK)
    logger.info(
        "Sizing: equity=$%.0f existing_gross=$%.0f risk_f=%.3f "
        "exposure_budget=$%.0f per_trade_cap=$%.0f api_room=$%.0f",
        equity,
        gross,
        risk_f,
        exposure_budget,
        per_trade_cap,
        remaining_api_room,
    )

    # ── Pass 1: forecast + provisionally decide for EVERY market ──
    # The provisional decision is unconstrained by remaining gross room
    # or the per-tick trade cap; it exists only to score each market's
    # edge (EV per share) so Pass 2 can rank by it.
    evaluated: list[dict] = []
    for m in markets:
        bid = float(m.quote.best_bid)
        ask = float(m.quote.best_ask)
        prior = max(0.01, min(0.99, (bid + ask) / 2.0))  # market mid = Bayesian prior

        # Regime POMDP belief update (per market). Use last-tick prior to
        # form a price-change observation; cold-start = uniform.
        regime_entropy = 0.0
        regime_state = "n/a"
        if cfg.regime_enabled:
            pomdp = regime_states.get(m.market_id) or RegimePOMDP()
            if pomdp.last_prior is not None and pomdp.last_prior > 0:
                delta = (prior - pomdp.last_prior) / pomdp.last_prior
                pomdp.update_from_change(delta, cfg.regime_change_bins)
            pomdp.last_prior = prior
            regime_states[m.market_id] = pomdp
            regime_entropy = pomdp.entropy()
            regime_state = pomdp.most_likely()

        belief = temper(
            prior,
            forecasts.get(m.market_id, []),
            ctrl.alpha,  # live llm_weight, tuned by the price-convergence controller
            cfg.llm_temperature,
            regime_entropy=regime_entropy,
            regime_weight=cfg.regime_weight,
        )
        rec = {
            "market_id": m.market_id,
            "family": getattr(m, "family", None),
            "question": m.question,
            "prior": round(prior, 4),
            "p_llm": round(belief.p_llm, 4),
            "p_yes": round(belief.p_yes, 4),
            "regime": regime_state,
            "regime_entropy": round(regime_entropy, 3),
            "action": "HOLD",
        }
        prov = decide(
            m.market_id, belief, bid, ask, bankroll, cfg,
            per_trade_cap=per_trade_cap,
            held_side=held.get(m.market_id), dd_state=dd_state,
        )
        evaluated.append({"m": m, "bid": bid, "ask": ask,
                          "belief": belief, "rec": rec, "prov": prov})

    # ── Exit pass: close held positions whose forecast edge is gone ──
    exit_intents: list[TradeIntentRequest] = []
    exiting: set[str] = set()
    if cfg.exit_enabled and portfolio:
        by_market = {e["m"].market_id: e for e in evaluated}
        for pos in portfolio.positions:
            if len(exit_intents) >= cfg.max_exits_per_tick:
                break
            ev = by_market.get(pos.market_id)
            if ev is None:
                continue  # not in this tick's snapshot — no fresh forecast
            x = decide_exit(pos.market_id, ev["belief"], pos.side,
                            float(pos.shares), float(pos.avg_entry_price),
                            float(pos.current_price), cfg)
            if x is None:
                continue
            exit_intents.append(TradeIntentRequest(
                market_id=x.market_id, action="SELL", side=x.side,
                shares=str(x.shares), idempotency_key=""))
            exiting.add(x.market_id)
            held.pop(x.market_id, None)
            ev["rec"].update(action=f"SELL {x.side}", shares=x.shares)
            logger.info("  EXIT %s %s x%d — %s",
                        x.side, x.market_id, x.shares, x.reason)

    # ── Pass 2: rank tradable markets by edge, submit the strongest ──
    # With a hard per-tick trade cap, the cap must be spent on our best
    # edges — not on whichever markets happened to come first.
    tradable = sorted(
        (e for e in evaluated
         if e["prov"] is not None and e["m"].market_id not in exiting),
        key=lambda e: e["prov"].edge, reverse=True,
    )
    buy_cap = max(0, trade_cap - len(exit_intents))
    logger.info("Evaluated %d markets — %d clear the EV threshold; "
                "submitting top %d by edge",
                len(evaluated), len(tradable), min(len(tradable), trade_cap))

    intents: list[TradeIntentRequest] = []
    planned_exposure = 0.0
    for e in tradable:
        remaining_budget = exposure_budget - planned_exposure
        if len(intents) >= buy_cap or remaining_budget <= 0:
            break
        m, rec = e["m"], e["rec"]
        opening_new = m.market_id not in held
        if opening_new and len(held) >= ruleset.MAX_OPEN_POSITIONS:
            rec["action"] = "HOLD (position cap)"
            continue
        # Re-size against the capital still unspent this tick.
        d = decide(
            m.market_id, e["belief"], e["bid"], e["ask"], bankroll, cfg,
            per_trade_cap=per_trade_cap,
            held_side=held.get(m.market_id), gross_room=remaining_budget,
            dd_state=dd_state,
        )
        if d is None:
            continue
        intents.append(TradeIntentRequest(
            market_id=d.market_id, action=d.action, side=d.side,
            shares=str(d.shares), idempotency_key="",
        ))
        planned_exposure += d.shares * d.price
        if opening_new:
            held[m.market_id] = d.side
        rec.update(action=f"BUY {d.side}", shares=d.shares,
                   price=round(d.price, 4), edge=round(d.edge, 4),
                   rank=len(intents))
        logger.info("  #%d  %s", len(intents), d.rationale)
    logger.info(
        "Planned %d trades, planned_exposure=$%.0f, total_if_filled=$%.0f (api_cap=$%.0f)",
        len(intents),
        planned_exposure,
        gross + planned_exposure,
        ruleset.MAX_GROSS_EXPOSURE,
    )

    audit: list[dict] = [e["rec"] for e in evaluated]

    session.put_plan(lease, idx, {"backend": cfg.resolved_backend(), "forecasts": audit})

    all_intents = exit_intents + intents
    if all_intents:
        result = session.submit_intents(lease, idx, all_intents)
        logger.info("Submitted %d (%d exits, %d buys) — %d ok, %d rejected",
                    len(all_intents), len(exit_intents), len(intents),
                    result.accepted, result.rejected)
        for r in result.rejections:
            logger.warning("  rejected %s: %s", r.intent_id, r.reason)
    else:
        logger.info("No exits or trades this tick.")

    session.finalize(lease, idx)
    session.complete_tick(lease)

    # Persist regime + peak state after each tick (idempotent).
    if cfg.regime_enabled:
        save_states(_REGIME_STATE_PATH, regime_states)
    _save_equity_peak(slug, peak)

    # ── Adaptive llm_weight controller ──
    # Log this tick's raw forecasts, then run a controller step if one is
    # due (~daily). The controller tunes ctrl.alpha — the live llm_weight
    # — toward how well p_llm predicts market price drift.
    ctrl.tick_count += 1
    controller.log_forecasts(_FORECAST_LOG_PATH, lease.tick_id, audit)
    controller.maybe_step(ctrl, _FORECAST_LOG_PATH, cfg.n_ticks)
    controller.save_state(_CONTROLLER_STATE_PATH, ctrl)


def run(cfg: Config) -> None:
    if not cfg.pa_api_key:
        raise SystemExit(
            "PA_SERVER_API_KEY is not set — cannot trade live.\n"
            "  • Get a key from the Prophet Arena operator, add it to .env\n"
            "  • Or run the offline harness (no key needed):\n"
            "        python -m prophet_arena.harness"
        )

    forecaster = JointForecaster(cfg)
    logger.info("Forecaster: family-aware joint (backend=%s, model=%s)",
                cfg.resolved_backend(), cfg.llm_model)
    logger.info(
        "Tempering: llm_weight=%.2f llm_temperature=%.3f (source=%s)",
        cfg.llm_weight, cfg.llm_temperature, cfg.llm_temperature_source,
    )
    logger.info(
        "Risk: dd_max=%.2f  Regime POMDP: %s (weight=%.2f)  Prune samples: %s",
        cfg.dd_max,
        "enabled" if cfg.regime_enabled else "disabled",
        cfg.regime_weight,
        "on" if cfg.prune_samples else "off",
    )

    # Load persisted state.
    regime_states = (
        load_states(_REGIME_STATE_PATH) if cfg.regime_enabled else {}
    )
    if cfg.regime_enabled and regime_states:
        evicted = evict_stale(regime_states, cfg.regime_max_age_hours)
        logger.info(
            "Loaded %d regime POMDPs (%d stale evicted)",
            len(regime_states), evicted,
        )
    equity_peak: dict[str, float] = {}
    ctrl = controller.load_state(_CONTROLLER_STATE_PATH, cfg.llm_weight)
    logger.info(
        "Adaptive llm_weight controller: alpha=%.3f (resumed at tick %d, %d prior steps)",
        ctrl.alpha, ctrl.tick_count, len(ctrl.history),
    )

    api = ServerAPIClient(base_url=cfg.pa_server_url, api_key=cfg.pa_api_key, timeout=30)
    with BenchmarkSession(api) as session:
        exp = session.create_experiment(
            slug=cfg.slug,
            config_hash=cfg.config_hash(),
            config_json=cfg.config_json(),
            n_ticks=cfg.n_ticks,
        )
        logger.info("Experiment %s (slug=%s, n_ticks=%d)",
                    exp.experiment_id, cfg.slug, cfg.n_ticks)
        part = session.upsert_participant(model=cfg.model_label,
                                          starting_cash=cfg.starting_cash)
        idx = getattr(part, "participant_idx", 0)
        equity_peak[cfg.slug] = max(_load_equity_peak(cfg.slug), cfg.starting_cash)
        logger.info(
            "Participant idx=%s, starting cash $%.0f, peak equity $%.0f",
            idx, cfg.starting_cash, equity_peak[cfg.slug],
        )

        while True:
            try:
                lease = session.claim_tick()
            except Exception as e:
                logger.warning("claim_tick failed: %s — retrying in 15s", e)
                time.sleep(15)
                continue

            if not lease.available:
                if lease.reason == "experiment_completed":
                    logger.info("Experiment completed.")
                    break
                wait = lease.retry_after_sec or 30
                logger.info("No tick available (%s) — sleeping %ss", lease.reason, wait)
                time.sleep(wait)
                continue

            try:
                _run_tick(session, cfg, forecaster, lease, idx, regime_states,
                          equity_peak, ctrl)
            except Exception as e:
                logger.error("Tick %s failed: %s", lease.tick_id, e, exc_info=True)
                try:
                    session.finalize(lease, idx, status="FAILED",
                                     error_code="agent_error", error_detail=str(e)[:200])
                    session.complete_tick(lease)
                except Exception as e2:
                    logger.error("tick cleanup failed: %s", e2)

        pf = session.get_portfolio(idx)
        if pf:
            logger.info("Final — cash $%s, equity $%s, total P&L $%s",
                        pf.cash, pf.equity, pf.total_pnl)


if __name__ == "__main__":
    run(load_config())
