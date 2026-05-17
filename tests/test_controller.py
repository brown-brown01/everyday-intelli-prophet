"""Tests for the price-convergence llm_weight controller (controller.py)."""
import json
from datetime import datetime, timedelta

from prophet_arena.bayesian_core import log_odds_to_prob, prob_to_log_odds
from prophet_arena.controller import (
    ALPHA_MAX,
    ALPHA_MIN,
    ControllerState,
    _ols_origin,
    estimate_beta,
    load_state,
    maybe_step,
    save_state,
)

_BASE = "2026-05-18T00:00:00+00:00"


def _write_log(path, n_markets: int, beta: float) -> None:
    """Write a synthetic forecast log where price drift d = beta * s exactly.

    Each market gets an anchor entry and a matched entry 25h later (past
    the 24h markout horizon), so estimate_beta should recover `beta`.
    """
    t0 = datetime.fromisoformat(_BASE)
    rows = []
    for i in range(n_markets):
        mid0 = 0.40 + 0.005 * (i % 20)        # 0.400 .. 0.495
        p_llm = 0.60 + 0.005 * (i % 20)       # always > mid0  -> positive s
        s = prob_to_log_odds(p_llm) - prob_to_log_odds(mid0)
        mid1 = log_odds_to_prob(prob_to_log_odds(mid0) + beta * s)
        fam = f"F{i % 8}"
        rows.append({"ts": t0.isoformat(), "market_id": f"M{i}",
                     "family": fam, "p_llm": p_llm, "mid": mid0})
        rows.append({"ts": (t0 + timedelta(hours=25)).isoformat(),
                     "market_id": f"M{i}", "family": fam,
                     "p_llm": p_llm, "mid": mid1})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ── OLS through the origin ──

def test_ols_origin_recovers_slope():
    obs = [(1.0, 0.5, "f"), (2.0, 1.0, "f"), (-1.0, -0.5, "f"), (3.0, 1.5, "f")]
    assert abs(_ols_origin(obs) - 0.5) < 1e-9


def test_ols_origin_zero_when_no_variance():
    assert _ols_origin([(0.0, 1.0, "f"), (0.0, -1.0, "f")]) == 0.0


# ── beta estimation from the forecast log ──

def test_estimate_beta_recovers_coefficient(tmp_path):
    log = tmp_path / "fl.jsonl"
    _write_log(log, 80, beta=0.5)
    est = estimate_beta(log, horizon_ticks=96)
    assert est["n"] == 80                       # one non-overlapping markout each
    assert abs(est["beta"] - 0.5) < 1e-6
    assert est["ci_low"] > 0.0                  # CI excludes zero


def test_estimate_beta_empty_log(tmp_path):
    est = estimate_beta(tmp_path / "missing.jsonl")
    assert est == {"beta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}


# ── controller step ──

def test_maybe_step_not_due():
    st = ControllerState(alpha=0.7, tick_count=50, last_step_tick=0)
    assert maybe_step(st, "unused.jsonl", n_ticks=1000) is None


def test_maybe_step_holds_on_insufficient_data(tmp_path):
    log = tmp_path / "fl.jsonl"
    _write_log(log, 10, beta=0.5)               # 10 obs < MIN_OBS
    st = ControllerState(alpha=0.70, tick_count=96, last_step_tick=0)
    rec = maybe_step(st, log, n_ticks=1000)
    assert rec is not None and "hold" in rec["action"]
    assert st.alpha == 0.70                     # unchanged


def test_maybe_step_updates_alpha_toward_beta(tmp_path):
    log = tmp_path / "fl.jsonl"
    _write_log(log, 80, beta=0.6)
    st = ControllerState(alpha=0.70, tick_count=96, last_step_tick=0)
    rec = maybe_step(st, log, n_ticks=1000)
    assert rec is not None and "updated" in rec["action"]
    # target = clamp(0.6) = 0.6 ;  alpha = 0.75*0.70 + 0.25*0.6 = 0.675
    assert abs(st.alpha - 0.675) < 1e-6
    assert st.last_step_tick == 96


def test_maybe_step_freezes_in_final_stretch(tmp_path):
    log = tmp_path / "fl.jsonl"
    _write_log(log, 80, beta=0.6)
    st = ControllerState(alpha=0.70, tick_count=900, last_step_tick=0)
    rec = maybe_step(st, log, n_ticks=1000)     # 1000-384=616; 900>616 -> frozen
    assert "frozen" in rec["action"]
    assert st.alpha == 0.70


def test_maybe_step_respects_alpha_bounds(tmp_path):
    log = tmp_path / "fl.jsonl"
    _write_log(log, 80, beta=5.0)               # wild beta -> target clamps to cap
    st = ControllerState(alpha=0.84, tick_count=96, last_step_tick=0)
    maybe_step(st, log, n_ticks=1000)
    assert ALPHA_MIN <= st.alpha <= ALPHA_MAX


# ── state persistence ──

def test_state_roundtrip(tmp_path):
    p = tmp_path / "cs.json"
    save_state(p, ControllerState(alpha=0.66, tick_count=123,
                                  last_step_tick=96, history=[{"a": 1}]))
    st = load_state(p, default_alpha=0.7)
    assert st.alpha == 0.66 and st.tick_count == 123 and st.last_step_tick == 96
    assert st.history == [{"a": 1}]


def test_load_state_default_when_missing(tmp_path):
    st = load_state(tmp_path / "nope.json", default_alpha=0.7)
    assert st.alpha == 0.7 and st.tick_count == 0 and st.history == []
