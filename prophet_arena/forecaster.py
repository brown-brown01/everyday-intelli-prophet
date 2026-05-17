"""LLM + heuristic forecaster for Prophet Arena markets.

Provider-agnostic: talks to any OpenAI-compatible chat endpoint
(OpenRouter by default — set ``LLM_BASE_URL`` / ``LLM_MODEL`` to switch).
When no LLM key is configured it falls back to a deterministic
base-rate heuristic so the pipeline always runs. LLM forecasts are
disk-cached, so the offline harness re-runs for free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .bayes import Forecast
from .config import Config, DATA_DIR

logger = logging.getLogger("prophet.forecaster")

_CACHE_PATH = DATA_DIR / "forecast_cache.json"
_CACHE_LOCK = threading.Lock()  # serializes forecast-cache disk writes across threads


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _CACHE_LOCK:
            _CACHE_PATH.write_text(json.dumps(cache))
    except Exception as e:  # pragma: no cover
        logger.debug("forecast cache write failed: %s", e)


def _clamp(p: float) -> float:
    return float(min(max(p, 0.02), 0.98))


@dataclass
class Question:
    """A normalized forecasting question (live market or sample event)."""

    id: str
    question: str
    rules: str = ""
    description: str = ""
    resolution_time: str = ""
    outcomes: list[str] = field(default_factory=list)
    target_outcome: str = ""  # multi-outcome: market resolves YES iff this wins

    def base_rate(self) -> float:
        n = len(self.outcomes)
        return 1.0 / n if n >= 2 else 0.5


SYSTEM_PROMPT = (
    "You are a calibrated superforecaster pricing prediction markets. "
    "Estimate the probability that the market resolves YES. Weigh base "
    "rates, concrete current evidence, and the time left until resolution. "
    "Avoid overconfidence. Respond with ONLY a JSON object of the form "
    '{"probability": <number between 0 and 1>, "reasoning": "<one or two sentences>"}.'
)


def _user_prompt(q: Question) -> str:
    lines = [f"Market question: {q.question}"]
    if q.rules:
        lines.append(f"Resolution rules (verbatim): {q.rules}")
    elif q.description:
        lines.append(f"Description: {q.description}")
    if q.target_outcome and q.outcomes:
        lines.append(
            f"This is a binary market: it resolves YES if and only if "
            f"'{q.target_outcome}' is the winning outcome, and NO otherwise. "
            f"The full candidate set is: {', '.join(q.outcomes)}."
        )
    if q.resolution_time:
        lines.append(f"Resolution time: {q.resolution_time}")
    lines.append(f"Today's date: {datetime.now(timezone.utc):%Y-%m-%d}.")
    lines.append('Return P(YES) as JSON: {"probability": ..., "reasoning": ...}')
    return "\n".join(lines)


def _parse(text: str) -> tuple[float, str]:
    """Pull a probability + reasoning out of an LLM response, defensively."""
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            raw = obj.get("probability", obj.get("p_yes", obj.get("p")))
            reason = str(obj.get("reasoning", obj.get("rationale", "")))[:280]
            return _clamp(float(raw)), reason
        except Exception:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return _clamp(float(m.group(1)) / 100.0), text[:180]
    m = re.search(r"0?\.\d+", text)
    if m:
        return _clamp(float(m.group(0))), text[:180]
    return 0.5, "unparseable: " + text[:160]


class LLMForecaster:
    """Forecaster backed by an OpenAI-compatible chat endpoint."""

    def __init__(self, cfg: Config):
        from openai import OpenAI

        self.cfg = cfg
        self.model = cfg.llm_model
        self.client = OpenAI(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            timeout=cfg.llm_timeout,
        )
        self._cache = _load_cache()

    def forecast(self, q: Question) -> list[Forecast]:
        out: list[Forecast] = []
        for i in range(max(1, self.cfg.llm_samples)):
            p, reason = self._one(q, i)
            out.append(Forecast(p_yes=p, source=f"llm#{i}", rationale=reason))
        if self.cfg.prune_samples and len(out) > 1:
            from .information import prune_redundant

            kept = prune_redundant(out, min_distance=self.cfg.prune_min_distance)
            if len(kept) < len(out):
                logger.info(
                    "Pruned %d/%d redundant samples (min_distance=%.3f) for %s",
                    len(out) - len(kept), len(out), self.cfg.prune_min_distance, q.id,
                )
            out = kept
        return out

    def _one(self, q: Question, idx: int) -> tuple[float, str]:
        prompt = _user_prompt(q)
        key = hashlib.sha256(
            f"{self.model}|{idx}|{SYSTEM_PROMPT}|{prompt}".encode()
        ).hexdigest()
        cached = self._cache.get(key)
        if cached:
            return cached["p"], cached.get("reason", "")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2 if idx == 0 else 0.8,
                max_tokens=320,
            )
            p, reason = _parse(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning("LLM forecast failed for %s: %s — using base rate", q.id, e)
            return q.base_rate(), f"llm-error: {e}"
        self._cache[key] = {"p": p, "reason": reason}
        _save_cache(self._cache)
        return p, reason


class HeuristicForecaster:
    """Deterministic no-LLM fallback.

    Returns the uniform base rate with a mild tilt toward the market's
    named (first-listed) outcome — the Prophet Arena sample events order
    candidates roughly by likelihood. Enough to exercise the full
    pipeline without an LLM key; not a competitive strategy.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def forecast(self, q: Question) -> list[Forecast]:
        base = q.base_rate()
        p = min(0.85, base * 1.6) if len(q.outcomes) > 2 else base
        return [Forecast(p_yes=p, source="heuristic",
                         rationale="base rate + favorite tilt")]


def make_forecaster(cfg: Config):
    """Return the forecaster backend selected by config."""
    backend = cfg.resolved_backend()
    if backend == "llm":
        if not cfg.llm_api_key:
            logger.warning("backend=llm but no LLM key set; using heuristic")
            return HeuristicForecaster(cfg)
        return LLMForecaster(cfg)
    return HeuristicForecaster(cfg)
