"""Offline scoring harness for the Prophet Arena forecasting brain.

Runs the Bayesian-tempered forecaster over a resolved sample dataset
and reports calibration (Brier score, log loss, accuracy) plus a
synthetic trading P&L against a uniform-prior market. Needs no Prophet
Arena key — only an LLM key for real forecasts, otherwise the
deterministic heuristic backend is used.

    python -m prophet_arena.harness [events.json]

Pull a dataset first, e.g.:
    prophet forecast retrieve --dataset sample-resolved --include-resolved \\
        -o data/prophet/resolved.json
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

from .bayes import temper
from .calibration import CalibrationRecord, fit_and_save
from .config import CALIBRATION_PATH, DATA_DIR, Config, load_config
from .forecaster import Question, make_forecaster
from .information import ensemble_diversity
from .strategy import decide

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

SPREAD = 0.04  # synthetic market spread used by the trading sim


def _yes_outcome(event: dict) -> str:
    """The outcome the binary market resolves YES on (first-listed)."""
    outs = event.get("outcomes") or []
    return outs[0] if outs else "Yes"


def _resolved_yes(event: dict):
    """True/False if the event is resolved, else None."""
    ro = event.get("resolved_outcome")
    if not ro:
        return None
    return _yes_outcome(event) in (ro.get("value") or [])


def _question(event: dict) -> Question:
    outs = event.get("outcomes") or []
    target = outs[0] if len(outs) >= 2 and outs[0].lower() not in ("yes", "no") else ""
    return Question(
        id=event.get("event_ticker") or event.get("title", "?"),
        question=event.get("title", ""),
        rules=event.get("rules") or "",
        description=event.get("description") or "",
        resolution_time=event.get("close_time") or "",
        outcomes=outs,
        target_outcome=target,
    )


def run(path: Path, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    events = json.loads(Path(path).read_text())
    resolved = [e for e in events if _resolved_yes(e) is not None]
    forecaster = make_forecaster(cfg)
    backend = cfg.resolved_backend()

    print("\n" + "=" * 74)
    print("  Prophet Arena — offline forecasting-brain harness")
    print("=" * 74)
    print(f"  dataset : {path}  ({len(resolved)}/{len(events)} resolved)")
    if backend == "llm":
        print(f"  backend : llm   model={cfg.llm_model}")
    else:
        print("  backend : heuristic   (set OPENROUTER_API_KEY for real LLM forecasts)")
    print(f"  temper  : llm_weight={cfg.llm_weight}  llm_temperature={cfg.llm_temperature}")
    print("=" * 74 + "\n")

    rows = []
    for i, e in enumerate(resolved, 1):
        q = _question(e)
        prior = q.base_rate()
        forecasts = forecaster.forecast(q)
        belief = temper(prior, forecasts, cfg.llm_weight, cfg.llm_temperature)
        rows.append({
            "q": q,
            "belief": belief,
            "prior": prior,
            "y": _resolved_yes(e),
            "diversity": ensemble_diversity(forecasts),
        })
        if backend == "llm":
            print(f"  [{i}/{len(resolved)}] forecast: {q.question[:54]}")

    metrics = _report(rows, cfg)

    if cfg.calibrate_after_harness and rows:
        records = [
            CalibrationRecord(
                p_raw=r["belief"].p_llm,
                p_tempered=r["belief"].p_yes,
                outcome=bool(r["y"]),
            )
            for r in rows
        ]
        result = fit_and_save(records, CALIBRATION_PATH)
        print(
            f"  Calibrator    β={result.beta:.3f}  "
            f"llm_temperature={result.llm_temperature:.3f}  "
            f"Brier {result.brier_before:.4f} → {result.brier_after:.4f}  "
            f"(n={result.n_records})  → {CALIBRATION_PATH}"
        )
        metrics["calibration"] = result.to_dict()

    return metrics


def _report(rows: list[dict], cfg: Config) -> dict:
    n = len(rows)
    if not n:
        print("No resolved events to score.")
        return {}

    brier = sum((r["belief"].p_yes - r["y"]) ** 2 for r in rows) / n
    brier_prior = sum((r["prior"] - r["y"]) ** 2 for r in rows) / n
    brier_coin = sum((0.5 - r["y"]) ** 2 for r in rows) / n

    def _ll(p: float, y: bool) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    logloss = sum(_ll(r["belief"].p_yes, r["y"]) for r in rows) / n
    acc = sum((r["belief"].p_yes > 0.5) == bool(r["y"]) for r in rows) / n

    # ── synthetic trading sim against a uniform-prior market ──
    staked = pnl = 0.0
    trades = wins = 0
    per_trade_cap = cfg.starting_cash * cfg.per_trade_risk_cap
    for r in rows:
        prior = r["prior"]
        bid = max(0.02, min(0.98, prior - SPREAD / 2))
        ask = max(0.02, min(0.98, prior + SPREAD / 2))
        d = decide(
            r["q"].id, r["belief"], bid, ask, cfg.starting_cash, cfg,
            per_trade_cap=per_trade_cap,
        )
        if not d:
            continue
        trades += 1
        cost = d.shares * d.price
        won = (d.side == "YES" and r["y"]) or (d.side == "NO" and not r["y"])
        payoff = d.shares * (1.0 if won else 0.0)
        staked += cost
        pnl += payoff - cost
        wins += 1 if payoff - cost > 0 else 0

    # ── per-event table ──
    print(f"  {'event':<46}{'prior':>7}{'p_llm':>7}{'p_yes':>7}{'res':>6}")
    print("  " + "-" * 71)
    for r in rows:
        b = r["belief"]
        hit = "ok" if (b.p_yes > 0.5) == bool(r["y"]) else "miss"
        print(f"  {r['q'].question[:44]:<46}{r['prior']:>7.3f}"
              f"{b.p_llm:>7.3f}{b.p_yes:>7.3f}{'YES' if r['y'] else 'NO':>6}  {hit}")

    skill = (1 - brier / brier_prior) * 100 if brier_prior else 0.0
    diversities = [float(r.get("diversity", 0.0)) for r in rows]
    mean_div = sum(diversities) / len(diversities) if diversities else 0.0
    print("\n  Forecast quality")
    print(f"    Brier score     {brier:.4f}   (base-rate prior {brier_prior:.4f} | "
          f"coin-flip {brier_coin:.4f})")
    print(f"    Log loss        {logloss:.4f}")
    print(f"    Accuracy        {acc * 100:.1f}%   over {n} resolved events")
    print(f"    Skill vs prior  {skill:+.1f}%   (positive = beats the base rate)")
    print(f"    Ensemble diversity  {mean_div:.4f}   "
          f"(mean pairwise |Δp| across LLM samples; 0 = mode-collapse)")

    print("\n  Synthetic trading sim  (uniform-prior market, 4c spread)")
    if trades:
        roi = pnl / staked * 100 if staked else 0.0
        print(f"    Trades taken    {trades} / {n}")
        print(f"    Hit rate        {wins / trades * 100:.1f}%")
        print(f"    Capital staked  ${staked:,.0f}")
        print(f"    Net P&L         ${pnl:+,.0f}   (ROI {roi:+.1f}%)")
    else:
        print(f"    No trades cleared the EV threshold (ev_threshold={cfg.ev_threshold}).")
    print()

    return {
        "n": n, "brier": brier, "brier_prior": brier_prior, "logloss": logloss,
        "accuracy": acc, "skill_vs_prior": skill, "trades": trades, "pnl": pnl,
        "ensemble_diversity": mean_div,
    }


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "resolved.json")
    run(Path(arg))
