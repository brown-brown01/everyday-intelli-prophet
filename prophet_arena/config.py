"""Configuration for the Prophet Arena hackathon bot.

Every value has a usable default so the offline harness runs with zero
setup. Live trading needs ``PA_SERVER_API_KEY``; the LLM forecaster
needs an OpenRouter (or other OpenAI-compatible) key. Values are read
from the environment / ``.env``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:  # .env is convenient but not required
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "prophet"


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_tuple_f(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    except ValueError:
        return default


@dataclass
class Config:
    # ── Prophet Arena server ──
    pa_server_url: str = "https://api.aiprophet.dev"
    pa_api_key: str = ""
    slug: str = "tradingbot-bayes-llm"
    model_label: str = "custom:tradingbot-bayes-llm"
    n_ticks: int = 96
    starting_cash: float = 10_000.0

    # ── LLM forecaster (any OpenAI-compatible endpoint) ──
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-4o-mini"
    forecaster_backend: str = "auto"  # auto | llm | heuristic
    llm_samples: int = 1
    llm_timeout: float = 40.0

    # ── Bayesian tempering ──
    llm_weight: float = 0.7       # opinion-pool weight on the LLM vs the prior
    llm_temperature: float = 1.0  # >1 flattens the LLM toward 0.5 (overconfidence fix)

    # ── Trading / sizing ──
    ev_threshold: float = 0.05
    kelly_fraction: float = 0.25
    max_notional_per_market: float = 1_000.0
    max_trades_per_tick: int = 20
    min_shares: int = 1

    # ── Tier-1 math additions ──
    prune_samples: bool = False                              # PRUNE_SAMPLES
    prune_min_distance: float = 0.02                         # PRUNE_MIN_DISTANCE
    regime_enabled: bool = False                             # REGIME_ENABLED
    regime_weight: float = 0.5                               # REGIME_WEIGHT
    regime_change_bins: tuple[float, ...] = (-0.05, -0.01, 0.01, 0.05)  # REGIME_CHANGE_BINS
    regime_max_age_hours: float = 24.0                       # REGIME_MAX_AGE_HOURS
    dd_max: float = 0.25                                     # DD_MAX
    calibrate_after_harness: bool = True                     # CALIBRATE_AFTER_HARNESS
    llm_temperature_source: str = "config"                   # set by load_config()

    def resolved_backend(self) -> str:
        """Which forecaster backend will actually be used."""
        if self.forecaster_backend in ("llm", "heuristic"):
            return self.forecaster_backend
        return "llm" if self.llm_api_key else "heuristic"

    def config_json(self) -> dict:
        """Audit blob persisted with the Prophet Arena experiment."""
        return {
            "strategy": "bayesian-tempered-llm",
            "version": "1.1",
            "llm_model": self.llm_model,
            "llm_weight": self.llm_weight,
            "llm_temperature": self.llm_temperature,
            "llm_temperature_source": self.llm_temperature_source,
            "ev_threshold": self.ev_threshold,
            "kelly_fraction": self.kelly_fraction,
            "prune_samples": self.prune_samples,
            "prune_min_distance": self.prune_min_distance,
            "regime_enabled": self.regime_enabled,
            "regime_weight": self.regime_weight,
            "regime_change_bins": list(self.regime_change_bins),
            "regime_max_age_hours": self.regime_max_age_hours,
            "dd_max": self.dd_max,
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.config_json(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


CALIBRATION_PATH = DATA_DIR / "calibration.json"


def _resolve_temperature(default: float = 1.0) -> tuple[float, str]:
    """Return (llm_temperature, source).

    ``LLM_TEMPERATURE`` env wins outright. Otherwise, if
    ``data/prophet/calibration.json`` exists, use the fitted value.
    Otherwise, fall back to ``default``.
    """
    if os.environ.get("LLM_TEMPERATURE", "") != "":
        return _env_f("LLM_TEMPERATURE", default), "env"
    try:
        raw = json.loads(CALIBRATION_PATH.read_text())
        return float(raw["llm_temperature"]), "calibration"
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default, "default"


def load_config() -> Config:
    """Build a :class:`Config` from environment variables / ``.env``."""
    llm_temp, llm_temp_source = _resolve_temperature(1.0)
    return Config(
        pa_server_url=os.getenv("PA_SERVER_URL", "https://api.aiprophet.dev"),
        pa_api_key=os.getenv("PA_SERVER_API_KEY", ""),
        slug=os.getenv("PA_SLUG", "tradingbot-bayes-llm"),
        model_label=os.getenv("PA_MODEL_LABEL", "custom:tradingbot-bayes-llm"),
        n_ticks=_env_i("PA_N_TICKS", 96),
        starting_cash=_env_f("PA_STARTING_CASH", 10_000.0),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_api_key=os.getenv("OPENROUTER_API_KEY", "") or os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        forecaster_backend=os.getenv("FORECASTER_BACKEND", "auto"),
        llm_samples=_env_i("LLM_SAMPLES", 1),
        llm_timeout=_env_f("LLM_TIMEOUT", 40.0),
        llm_weight=_env_f("LLM_WEIGHT", 0.7),
        llm_temperature=llm_temp,
        ev_threshold=_env_f("EV_THRESHOLD", 0.05),
        kelly_fraction=_env_f("KELLY_FRACTION", 0.25),
        max_notional_per_market=_env_f("MAX_NOTIONAL_PER_MARKET", 1_000.0),
        max_trades_per_tick=_env_i("MAX_TRADES_PER_TICK", 20),
        min_shares=_env_i("MIN_SHARES", 1),
        prune_samples=_env_b("PRUNE_SAMPLES", False),
        prune_min_distance=_env_f("PRUNE_MIN_DISTANCE", 0.02),
        regime_enabled=_env_b("REGIME_ENABLED", False),
        regime_weight=_env_f("REGIME_WEIGHT", 0.5),
        regime_change_bins=_env_tuple_f("REGIME_CHANGE_BINS", (-0.05, -0.01, 0.01, 0.05)),
        regime_max_age_hours=_env_f("REGIME_MAX_AGE_HOURS", 24.0),
        dd_max=_env_f("DD_MAX", 0.25),
        calibrate_after_harness=_env_b("CALIBRATE_AFTER_HARNESS", True),
        llm_temperature_source=llm_temp_source,
    )
