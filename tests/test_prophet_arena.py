"""Unit tests for Prophet Arena Tier-1 mathematical additions."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from prophet_arena.bayes import (
    Belief,
    Forecast,
    geometric_mean_pool,
    log_odds_pool,
    temper,
)
from prophet_arena.calibration import (
    CalibrationRecord,
    TemperatureCalibrator,
    apply_beta,
    brier,
    fit_and_save,
    load,
    save,
)
from prophet_arena.config import Config
from prophet_arena.information import (
    binned_mutual_information,
    cross_sample_redundancy,
    ensemble_diversity,
    entropy,
    prune_redundant,
)
from prophet_arena.pomdp import (
    OBSERVATIONS,
    STATES,
    RegimePOMDP,
    discretize_change,
    evict_stale,
    load_states,
    save_states,
)
from prophet_arena.strategy import (
    DrawdownState,
    decide,
    kelly_drawdown_adjustment,
)


def _cfg(**kw) -> Config:
    """Build a default Config for tests, allowing overrides."""
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pool helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestPoolHelpers:
    def test_geometric_mean_pool_identity(self):
        assert geometric_mean_pool([0.5, 0.5, 0.5]) == pytest.approx(0.5, abs=1e-6)

    def test_geometric_mean_pool_known(self):
        # (0.2 * 0.8)^(1/2) = 0.4
        assert geometric_mean_pool([0.2, 0.8]) == pytest.approx(0.4, abs=1e-6)

    def test_geometric_mean_pool_empty(self):
        assert geometric_mean_pool([]) == 0.5

    def test_log_odds_pool_equal_weight_matches_inline(self):
        # The pre-refactor inline used np.mean of logits — confirm parity.
        probs = [0.3, 0.6, 0.7]
        from prophet_arena.bayesian_core import log_odds_to_prob, prob_to_log_odds

        manual = log_odds_to_prob(float(np.mean([prob_to_log_odds(p) for p in probs])))
        assert log_odds_pool(probs) == pytest.approx(manual, abs=1e-9)

    def test_log_odds_pool_weights_prefer_heavy(self):
        # All-weight on the high sample collapses to the high sample.
        assert log_odds_pool([0.1, 0.9], weights=[0.0, 1.0]) == pytest.approx(0.9, abs=1e-6)
        assert log_odds_pool([0.1, 0.9], weights=[1.0, 0.0]) == pytest.approx(0.1, abs=1e-6)

    def test_log_odds_pool_weights_shape_check(self):
        with pytest.raises(ValueError):
            log_odds_pool([0.3, 0.6], weights=[1.0])

    def test_log_odds_pool_zero_weights_fall_back_to_mean(self):
        probs = [0.3, 0.6]
        assert log_odds_pool(probs, weights=[0, 0]) == pytest.approx(
            log_odds_pool(probs), abs=1e-9
        )

    def test_temper_regime_entropy_inflates_uncertainty(self):
        fcs = [Forecast(0.7), Forecast(0.72), Forecast(0.69)]
        base = temper(0.5, fcs, llm_weight=0.7, regime_entropy=0.0)
        hot = temper(0.5, fcs, llm_weight=0.7, regime_entropy=1.0)
        assert hot.uncertainty > base.uncertainty

    def test_temper_sample_weights_change_pllm(self):
        fcs = [Forecast(0.2), Forecast(0.8)]
        weighted = temper(0.5, fcs, llm_weight=1.0, sample_weights=[0.0, 1.0])
        assert weighted.p_llm == pytest.approx(0.8, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RegimePOMDP
# ─────────────────────────────────────────────────────────────────────────────


class TestRegimePOMDP:
    def test_uniform_init_sums_to_one(self):
        p = RegimePOMDP()
        assert sum(p.belief) == pytest.approx(1.0, abs=1e-9)
        assert len(p.belief) == len(STATES)

    def test_entropy_uniform_is_ln_n(self):
        p = RegimePOMDP()
        assert p.entropy() == pytest.approx(math.log(len(STATES)), abs=1e-9)

    def test_update_preserves_sum_to_one(self):
        p = RegimePOMDP()
        for obs in ("up_big", "flat", "down_small", "up_big"):
            p.update(obs)
            assert sum(p.belief) == pytest.approx(1.0, abs=1e-9)

    def test_up_big_drift_toward_bull(self):
        p = RegimePOMDP()
        for _ in range(20):
            p.update("up_big")
        assert p.most_likely() == "bull"

    def test_down_big_drift_toward_bear(self):
        p = RegimePOMDP()
        for _ in range(20):
            p.update("down_big")
        assert p.most_likely() == "bear"

    def test_flat_drift_toward_sideways(self):
        p = RegimePOMDP()
        for _ in range(20):
            p.update("flat")
        assert p.most_likely() == "sideways"

    def test_unknown_observation_is_noop(self):
        p = RegimePOMDP()
        before = p.belief.copy()
        p.update("not_a_real_observation")
        assert np.array_equal(p.belief, before)
        assert p.n_updates == 0

    def test_round_trip_serialization(self):
        p = RegimePOMDP()
        p.update("up_big")
        p.last_prior = 0.42
        restored = RegimePOMDP.from_dict(p.to_dict())
        assert np.allclose(restored.belief, p.belief)
        assert restored.n_updates == p.n_updates
        assert restored.last_seen == p.last_seen
        assert restored.last_prior == pytest.approx(0.42)

    def test_discretize_change_bins(self):
        assert discretize_change(-0.10) == "down_big"
        assert discretize_change(-0.02) == "down_small"
        assert discretize_change(0.0) == "flat"
        assert discretize_change(0.02) == "up_small"
        assert discretize_change(0.10) == "up_big"

    def test_update_from_change_drift(self):
        p = RegimePOMDP()
        for _ in range(15):
            p.update_from_change(0.10)
        assert p.most_likely() == "bull"

    def test_save_load_round_trip(self, tmp_path):
        states = {
            "m1": RegimePOMDP(),
            "m2": RegimePOMDP(),
        }
        states["m1"].update("up_big")
        states["m2"].last_prior = 0.31
        path = tmp_path / "regime.json"
        save_states(path, states)
        loaded = load_states(path)
        assert set(loaded) == {"m1", "m2"}
        assert np.allclose(loaded["m1"].belief, states["m1"].belief)
        assert loaded["m2"].last_prior == pytest.approx(0.31)

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_states(tmp_path / "absent.json") == {}

    def test_evict_stale_drops_old(self):
        from datetime import datetime, timezone, timedelta

        states = {"m1": RegimePOMDP(), "m2": RegimePOMDP()}
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        states["m1"].last_seen = old
        removed = evict_stale(states, max_age_hours=24.0)
        assert removed == 1
        assert "m1" not in states
        assert "m2" in states


# ─────────────────────────────────────────────────────────────────────────────
# 3. Information theory
# ─────────────────────────────────────────────────────────────────────────────


class TestInformation:
    def test_binary_entropy_max_at_half(self):
        assert entropy(0.5) == pytest.approx(math.log(2), abs=1e-9)

    def test_binary_entropy_zero_at_extremes(self):
        assert entropy(0.0) == pytest.approx(0.0, abs=1e-6)
        assert entropy(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_categorical_entropy_uniform(self):
        assert entropy([0.25] * 4) == pytest.approx(math.log(4), abs=1e-9)

    def test_categorical_entropy_concentrated_is_zero(self):
        assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    def test_mi_independent_near_zero(self):
        rng = np.random.default_rng(7)
        xs = rng.uniform(0, 1, size=400).tolist()
        ys = rng.uniform(0, 1, size=400).tolist()
        mi = binned_mutual_information(xs, ys, bins=10)
        assert mi == pytest.approx(0.0, abs=0.25)

    def test_mi_identical_high(self):
        rng = np.random.default_rng(3)
        xs = rng.uniform(0, 1, size=400).tolist()
        mi = binned_mutual_information(xs, xs, bins=10)
        # Should be ≈ entropy of binned X, which is meaningfully positive.
        assert mi > 0.5

    def test_mi_symmetric(self):
        rng = np.random.default_rng(11)
        xs = rng.uniform(0, 1, size=200).tolist()
        ys = (np.asarray(xs) * 0.5 + rng.uniform(0, 0.5, size=200)).tolist()
        assert binned_mutual_information(xs, ys) == pytest.approx(
            binned_mutual_information(ys, xs), abs=1e-9
        )

    def test_ensemble_diversity_identical_is_zero(self):
        fcs = [Forecast(0.7), Forecast(0.7), Forecast(0.7)]
        assert ensemble_diversity(fcs) == pytest.approx(0.0, abs=1e-9)

    def test_ensemble_diversity_max_spread(self):
        fcs = [Forecast(0.0), Forecast(1.0)]
        assert ensemble_diversity(fcs) == pytest.approx(1.0, abs=1e-2)

    def test_prune_redundant_collapses_duplicates(self):
        fcs = [Forecast(0.5), Forecast(0.501), Forecast(0.502)]
        kept = prune_redundant(fcs, min_distance=0.02)
        assert len(kept) == 1

    def test_prune_redundant_keeps_distinct(self):
        fcs = [Forecast(0.2), Forecast(0.5), Forecast(0.8)]
        kept = prune_redundant(fcs, min_distance=0.02)
        assert len(kept) == 3

    def test_prune_redundant_min_distance_zero_is_noop(self):
        fcs = [Forecast(0.5), Forecast(0.5), Forecast(0.5)]
        assert len(prune_redundant(fcs, min_distance=0.0)) == 3

    def test_cross_sample_redundancy_shape(self):
        matrix = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.85]]
        m = cross_sample_redundancy(matrix, bins=4)
        assert len(m) == 2 and all(len(row) == 2 for row in m)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temperature calibration
# ─────────────────────────────────────────────────────────────────────────────


class TestCalibration:
    def test_apply_beta_identity(self):
        # β=1 leaves p unchanged.
        for p in (0.1, 0.4, 0.7, 0.95):
            assert apply_beta(p, 1.0) == pytest.approx(p, abs=1e-6)

    def test_apply_beta_flattens_when_low(self):
        # β<1 pulls toward 0.5.
        assert apply_beta(0.95, 0.5) < 0.95
        assert apply_beta(0.05, 0.5) > 0.05

    def test_apply_beta_sharpens_when_high(self):
        assert apply_beta(0.7, 5.0) > 0.7
        assert apply_beta(0.3, 5.0) < 0.3

    def test_calibrator_reduces_brier_on_overconfident(self):
        # Construct synthetic overconfident data: model says 0.95 / 0.05
        # but truth is 50/50. β<1 should help.
        records = [
            CalibrationRecord(p_raw=0.95, p_tempered=0.95, outcome=True),
            CalibrationRecord(p_raw=0.95, p_tempered=0.95, outcome=False),
            CalibrationRecord(p_raw=0.05, p_tempered=0.05, outcome=True),
            CalibrationRecord(p_raw=0.05, p_tempered=0.05, outcome=False),
        ] * 25
        res = TemperatureCalibrator().fit(records)
        assert res.brier_after <= res.brier_before + 1e-9
        assert res.beta < 1.0

    def test_calibrator_sharpens_underconfident(self):
        # Model always says 0.55 / 0.45 but is actually right when high.
        records = [
            CalibrationRecord(p_raw=0.55, p_tempered=0.55, outcome=True),
            CalibrationRecord(p_raw=0.45, p_tempered=0.45, outcome=False),
        ] * 50
        res = TemperatureCalibrator().fit(records)
        assert res.beta > 1.0

    def test_calibrator_empty_records_safe(self):
        res = TemperatureCalibrator().fit([])
        assert res.beta == 1.0
        assert res.llm_temperature == 1.0

    def test_save_load_round_trip(self, tmp_path):
        records = [
            CalibrationRecord(p_raw=0.6, p_tempered=0.6, outcome=True),
            CalibrationRecord(p_raw=0.4, p_tempered=0.4, outcome=False),
        ]
        path = tmp_path / "calibration.json"
        res = fit_and_save(records, path)
        restored = load(path)
        assert restored is not None
        assert restored.beta == pytest.approx(res.beta, abs=1e-9)
        assert restored.llm_temperature == pytest.approx(res.llm_temperature, abs=1e-9)
        assert restored.n_records == 2

    def test_load_missing_returns_none(self, tmp_path):
        assert load(tmp_path / "absent.json") is None

    def test_brier_matches_manual(self):
        records = [
            CalibrationRecord(p_raw=0.8, p_tempered=0.8, outcome=True),
            CalibrationRecord(p_raw=0.3, p_tempered=0.3, outcome=False),
        ]
        manual = ((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2.0
        assert brier(records) == pytest.approx(manual, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Drawdown-adjusted Kelly
# ─────────────────────────────────────────────────────────────────────────────


class TestDrawdownAdjustedKelly:
    def test_no_drawdown_full_size(self):
        assert kelly_drawdown_adjustment(0.0, 0.25) == pytest.approx(1.0)

    def test_max_drawdown_zero_size(self):
        assert kelly_drawdown_adjustment(0.25, 0.25) == pytest.approx(0.0)
        assert kelly_drawdown_adjustment(0.5, 0.25) == pytest.approx(0.0)

    def test_half_drawdown_half_size(self):
        assert kelly_drawdown_adjustment(0.125, 0.25) == pytest.approx(0.5)

    def test_disabled_when_dd_max_nonpositive(self):
        assert kelly_drawdown_adjustment(0.5, 0.0) == pytest.approx(1.0)
        assert kelly_drawdown_adjustment(0.5, -1.0) == pytest.approx(1.0)

    def _belief(self, p: float, uncertainty: float = 0.08) -> Belief:
        return Belief(p_yes=p, prior=0.5, p_llm=p, uncertainty=uncertainty,
                      n_forecasts=1, forecasts=[Forecast(p_yes=p)])

    def test_decide_with_dd_halves_shares_at_half_dd(self):
        cfg = _cfg(ev_threshold=0.0, kelly_fraction=1.0,
                   min_shares=1, dd_max=0.25)
        belief = self._belief(0.8, uncertainty=0.01)
        bid, ask, bankroll = 0.49, 0.50, 100_000.0
        d_full = decide("m1", belief, bid, ask, bankroll, cfg, per_trade_cap=1e9)
        d_half = decide("m1", belief, bid, ask, bankroll, cfg,
                        per_trade_cap=1e9,
                        dd_state=DrawdownState(current_dd=0.125, dd_max=0.25))
        d_zero = decide("m1", belief, bid, ask, bankroll, cfg,
                        per_trade_cap=1e9,
                        dd_state=DrawdownState(current_dd=0.25, dd_max=0.25))
        assert d_full is not None and d_half is not None
        assert d_zero is None  # 0 shares → no decision
        # Half-DD should give roughly half the shares of full-size.
        assert d_half.shares == pytest.approx(d_full.shares / 2, rel=0.05)

    def test_decide_dd_state_none_is_unchanged(self):
        cfg = _cfg(ev_threshold=0.0, kelly_fraction=0.25, min_shares=1)
        belief = self._belief(0.7)
        bid, ask, bankroll = 0.49, 0.50, 10_000.0
        d_a = decide("m", belief, bid, ask, bankroll, cfg, per_trade_cap=1e9)
        d_b = decide("m", belief, bid, ask, bankroll, cfg, per_trade_cap=1e9, dd_state=None)
        assert d_a is not None and d_b is not None
        assert d_a.shares == d_b.shares


# ─────────────────────────────────────────────────────────────────────────────
# 6. Config wiring
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigWiring:
    def test_config_hash_changes_with_regime_flag(self, monkeypatch):
        from prophet_arena.config import load_config

        monkeypatch.delenv("REGIME_ENABLED", raising=False)
        h_off = load_config().config_hash()

        monkeypatch.setenv("REGIME_ENABLED", "1")
        h_on = load_config().config_hash()
        assert h_off != h_on

    def test_config_hash_changes_with_dd_max(self, monkeypatch):
        from prophet_arena.config import load_config

        monkeypatch.delenv("DD_MAX", raising=False)
        h_a = load_config().config_hash()
        monkeypatch.setenv("DD_MAX", "0.5")
        h_b = load_config().config_hash()
        assert h_a != h_b

    def test_env_temperature_wins_over_calibration_file(self, monkeypatch, tmp_path):
        # Point calibration path at an isolated temp file containing β=2.
        from prophet_arena import config as cfg_mod

        fake = tmp_path / "calibration.json"
        fake.write_text(
            '{"beta": 2.0, "llm_temperature": 0.5, "n_records": 10, '
            '"brier_before": 0.3, "brier_after": 0.25, "fitted_at": "x"}'
        )
        monkeypatch.setattr(cfg_mod, "CALIBRATION_PATH", fake)
        monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
        c_no_env = cfg_mod.load_config()
        assert c_no_env.llm_temperature == pytest.approx(0.5)
        assert c_no_env.llm_temperature_source == "calibration"

        monkeypatch.setenv("LLM_TEMPERATURE", "3.0")
        c_env = cfg_mod.load_config()
        assert c_env.llm_temperature == pytest.approx(3.0)
        assert c_env.llm_temperature_source == "env"
