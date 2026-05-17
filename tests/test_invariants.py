"""Invariant tests — family coherence (#6) and risk caps (#8).

The two launch-critical invariants from the pre-launch checklist:
mutual-exclusivity families must produce coherent probabilities, and
position sizing must never breach the Prophet Arena hard caps.
"""
from types import SimpleNamespace

from prophet_arena.bayes import Belief, Forecast
from prophet_arena.config import Config
from prophet_arena.families import _parse_joint, check_coherence, group_by_family
from prophet_arena.strategy import DrawdownState, decide


def _mkt(market_id, family=None, topic=None):
    return SimpleNamespace(market_id=market_id, family=family, topic=topic,
                           question=f"Q {market_id}")


# ── #6 — family coherence ──────────────────────────────────────────────

def test_parse_joint_normalizes_to_one():
    probs = _parse_joint('{"probabilities": [2, 2, 4, 2]}', 4)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_parse_joint_clamps_negatives_then_normalizes():
    probs = _parse_joint('{"probabilities": [-1, 1, 1]}', 3)
    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p >= 0.0 for p in probs)


def test_parse_joint_rejects_wrong_length():
    assert _parse_joint('{"probabilities": [0.5, 0.5]}', 3) is None


def test_parse_joint_rejects_garbage_and_zero_sum():
    assert _parse_joint("not json at all", 3) is None
    assert _parse_joint('{"probabilities": [0, 0, 0]}', 3) is None


def test_parse_joint_ignores_surrounding_prose():
    probs = _parse_joint('Here is my answer:\n{"probabilities": [0.3, 0.7]}\nDone.', 2)
    assert probs is not None and abs(sum(probs) - 1.0) < 1e-9


def test_parse_joint_handles_reasoning_field():
    text = ('{"reasoning": "moved the favorite up; the rest stay near market", '
            '"probabilities": [0.6, 0.25, 0.15]}')
    probs = _parse_joint(text, 3)
    assert probs is not None and abs(probs[0] - 0.6) < 1e-9


def test_group_by_family_groups_and_isolates_singletons():
    markets = [_mkt("a", family="F1"), _mkt("b", family="F1"),
               _mkt("c", family="F2"), _mkt("d")]  # d -> singleton (own id)
    groups = group_by_family(markets)
    assert len(groups["F1"]) == 2
    assert len(groups["F2"]) == 1
    assert len(groups["d"]) == 1


def test_check_coherence_passes_clean_output():
    out = {"a": [Forecast(p_yes=0.3)], "b": [Forecast(p_yes=0.7)]}
    assert check_coherence(out) == []


def test_check_coherence_flags_out_of_range():
    out = {"a": [Forecast(p_yes=0.0)], "b": [Forecast(p_yes=1.5)],
           "c": [Forecast(p_yes=0.5)]}
    violations = check_coherence(out)
    assert len(violations) == 2
    assert any("a" in v for v in violations) and any("b" in v for v in violations)


# ── #8 — risk caps / sizing limits ─────────────────────────────────────

def _belief(p):
    return Belief(p_yes=p, prior=0.5, p_llm=p, uncertainty=0.06, n_forecasts=1)


def test_notional_never_exceeds_per_market_cap():
    cfg = Config()
    d = decide("m", _belief(0.97), best_bid=0.18, best_ask=0.20,
               bankroll=10_000_000, cfg=cfg)
    assert d is not None
    assert d.shares * d.price <= cfg.max_notional_per_market + 1e-6


def test_notional_never_exceeds_gross_room():
    d = decide("m", _belief(0.97), best_bid=0.18, best_ask=0.20,
               bankroll=10_000_000, cfg=Config(), gross_room=50.0)
    assert d is not None
    assert d.shares * d.price <= 50.0 + 1e-6


def test_drawdown_shrinks_position():
    cfg = Config()
    # bankroll small enough that neither hits the $1k notional cap,
    # so the drawdown multiplier is actually visible.
    flat = decide("m", _belief(0.95), 0.28, 0.30, 4_000, cfg,
                  dd_state=DrawdownState(current_dd=0.0, dd_max=0.25))
    bleeding = decide("m", _belief(0.95), 0.28, 0.30, 4_000, cfg,
                      dd_state=DrawdownState(current_dd=0.20, dd_max=0.25))
    assert flat is not None and bleeding is not None
    assert bleeding.shares < flat.shares


def test_tiny_bankroll_returns_none():
    assert decide("m", _belief(0.95), 0.28, 0.30, bankroll=0.10, cfg=Config()) is None


def test_decision_price_always_valid():
    cfg = Config()
    yes = decide("m", _belief(0.95), 0.40, 0.45, 100_000, cfg)
    no = decide("m", _belief(0.05), 0.55, 0.60, 100_000, cfg)
    for d in (yes, no):
        assert d is not None
        assert 0.0 < d.price < 1.0
        assert d.shares > 0
