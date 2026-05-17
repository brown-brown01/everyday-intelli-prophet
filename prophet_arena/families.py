"""Family-aware joint forecasting for Prophet Arena.

Prophet Arena markets that belong to one mutually-exclusive event share
a ``family`` id — e.g. all 28 "2028 Democratic nominee" markets are
family ``KXPRESNOMD``, and exactly one of them resolves YES. Forecasting
them independently produces probabilities that don't sum to 1, so the
strategy layer finds fake "edge" on every cheap longshot and piles into
mutually-exclusive bets.

This module forecasts each family in a *single* LLM call — the model
sees the whole slate at once — and normalizes the result so the family
is coherent (sums to 1). Singleton markets fall back to the per-market
forecaster. As a bonus this collapses ~256 per-market calls into ~one
call per family, which keeps a full tick inside the 9-minute deadline.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone

from .bayes import Forecast
from .config import Config, DATA_DIR
from .forecaster import Question, _clamp, make_forecaster

logger = logging.getLogger("prophet.families")

_CACHE_PATH = DATA_DIR / "family_cache.json"
_CACHE_LOCK = threading.Lock()  # serializes family-cache disk writes across threads

JOINT_SYSTEM = (
    "You are forecasting a set of MUTUALLY EXCLUSIVE outcomes for one event. "
    "Exactly one outcome resolves YES, so the true probabilities must sum to 1. "
    "Each outcome is shown with its current market-implied probability. "
    "Treat that market probability as the default prior — it aggregates public "
    "information reasonably well. Your job is to make conservative updates to "
    "that prior only when concrete, specific, resolution-relevant evidence or a "
    "strong structural/base-rate argument justifies it; you are updating a "
    "prior, not re-pricing the slate from scratch. "
    "If you do not have a strong reason to move an outcome, keep it close to "
    "the market. In wide fields most candidates should remain genuine longshots "
    "unless there is clear reason otherwise (long-dated markets tend to "
    "overprice longshots). Avoid extreme probabilities unless the rules or "
    "overwhelming public facts nearly determine the result. Prefer calibration "
    "over boldness — an unjustified confident number is worse than the market. "
    "Respond with ONLY valid JSON in this exact format: "
    '{"reasoning":"<2-4 brief sentences on the main deviations from market>",'
    '"probabilities":[<p1>,<p2>,...,<pN>]}'
)


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
        logger.debug("family cache write failed: %s", e)


def _question(m) -> Question:
    """Build a Question from a Prophet Arena MarketData."""
    return Question(
        id=m.market_id,
        question=m.question,
        description=getattr(m, "description", "") or "",
        resolution_time=str(getattr(m, "resolution_time", "") or ""),
    )


def group_by_family(markets) -> dict[str, list]:
    """Group markets by their mutual-exclusivity family id.

    Falls back to ``topic`` then ``market_id`` (a singleton) when
    ``family`` is missing.
    """
    groups: dict[str, list] = defaultdict(list)
    for m in markets:
        key = getattr(m, "family", None) or getattr(m, "topic", None) or m.market_id
        groups[key].append(m)
    return dict(groups)


def _joint_prompt(label: str, members: list) -> str:
    """Build the joint user prompt.

    ``members`` are Prophet Arena MarketData. Each line shows the outcome
    and its current market-implied probability (the quote mid) so the
    model anchors on the market as its prior instead of re-pricing blind.
    """
    lines = [
        f"Event: {label}",
        f"The {len(members)} outcomes below are mutually exclusive: "
        "exactly one resolves YES.",
        "Each line shows the outcome and its current market-implied probability.",
        "",
    ]
    for i, m in enumerate(members, 1):
        bid = float(m.quote.best_bid)
        ask = float(m.quote.best_ask)
        mid = max(0.01, min(0.99, (bid + ask) / 2.0))
        lines.append(f"{i}. {m.question} — market {mid:.4f}")
    lines += [
        "",
        f"Today's date: {datetime.now(timezone.utc):%Y-%m-%d}.",
        "Start from the market probabilities as your prior. Only deviate when "
        "you have a concrete, specific, resolution-relevant reason; if evidence "
        "is weak, mixed, stale, or absent, stay close to the market.",
        f'Return ONLY JSON with exactly {len(members)} probabilities, in order, '
        'summing to 1: {"reasoning":"...","probabilities":[...]}',
    ]
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Return the first balanced, parseable JSON object in ``text``.

    Scans brace depth (string-aware) so a ``reasoning`` field or prose
    around the JSON cannot derail the parse — unlike a greedy regex.
    """
    text = text or ""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break  # malformed — try the next '{'
        start = text.find("{", start + 1)
    return None


def _parse_joint(text: str, n: int) -> list[float] | None:
    """Extract an n-length probability vector from an LLM response.

    Takes the first valid JSON object — ignoring surrounding prose and
    any ``reasoning`` field — then renormalizes the vector to sum to 1.
    """
    obj = _extract_json(text)
    if obj is None:
        return None
    arr = obj.get("probabilities") or obj.get("probs") or obj.get("p")
    if not isinstance(arr, list) or len(arr) != n:
        return None
    try:
        vals = [max(0.0, float(x)) for x in arr]
    except (TypeError, ValueError):
        return None
    total = sum(vals)
    if total <= 0:
        return None
    return [v / total for v in vals]


def check_coherence(forecasts: dict[str, list[Forecast]]) -> list[str]:
    """Invariant check: every forecast probability must lie strictly in (0, 1).

    Returns a list of human-readable violations (empty when all clean).
    This bound is unambiguous — anything outside it is a bug, not clamp
    drift — so the tick loop logs any violation as a warning.
    """
    violations: list[str] = []
    for market_id, fcs in forecasts.items():
        for fc in fcs:
            if not (0.0 < fc.p_yes < 1.0):
                violations.append(f"{market_id}: p_yes={fc.p_yes!r} outside (0,1)")
    return violations


class JointForecaster:
    """Family-aware forecaster.

    ``forecast_markets(markets)`` groups the markets by family, forecasts
    each multi-market family in one normalized LLM call, and forecasts
    singletons with the per-market forecaster. Returns a
    ``market_id -> [Forecast]`` map the tick loop can consume directly.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = make_forecaster(cfg)  # singletons + joint fallback
        self._cache = _load_cache()
        self._client = None
        if cfg.resolved_backend() == "llm" and cfg.llm_api_key:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=cfg.llm_base_url,
                api_key=cfg.llm_api_key,
                timeout=cfg.llm_timeout,
            )

    def forecast_markets(self, markets) -> dict[str, list[Forecast]]:
        out: dict[str, list[Forecast]] = {}
        groups = group_by_family(markets)
        n_joint = sum(1 for v in groups.values() if len(v) > 1)
        logger.info(
            "%d markets -> %d families (%d joint, %d singleton)",
            len(markets), len(groups), n_joint, len(groups) - n_joint,
        )

        def _do(item: tuple) -> dict[str, list[Forecast]]:
            fam, members = item
            try:
                if len(members) < 2 or self._client is None:
                    return {m.market_id: self.base.forecast(_question(m))
                            for m in members}
                return self._joint(fam, members)
            except Exception as e:  # never let one family kill the tick
                logger.warning("family %s failed: %s", fam, e)
                return {}

        # Forecast families concurrently — each is one slow LLM(+search)
        # call. A cold tick has ~76; sequential they would blow the
        # 9-minute submission deadline. The calls are network-bound and
        # cache writes are lock-guarded, so a thread pool is safe.
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_do, list(groups.items())))
        for r in results:
            out.update(r)
        for v in check_coherence(out):
            logger.warning("coherence violation — %s", v)
        return out

    def _joint(self, fam: str, members: list) -> dict[str, list[Forecast]]:
        label = getattr(members[0], "topic", None) or fam
        prompt = _joint_prompt(label, members)

        # Cache key: family + membership + UTC day — deliberately NOT the
        # live mid-prices. The prompt embeds current prices (so the model
        # anchors on them), but prices move every 15-min tick; keying the
        # cache on them would re-run the LLM + web search every tick —
        # ~100x the cost for a forecast that barely changes. So: one joint
        # forecast per family per day. The tick loop still re-prices the
        # trade decision every tick against the live market mid.
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        member_ids = sorted(m.market_id for m in members)
        key = hashlib.sha256(
            f"{self.cfg.llm_model}|joint|{day}|{label}|{','.join(member_ids)}".encode()
        ).hexdigest()

        # Cache value is a {market_id: prob} map — order-independent, so a
        # change in snapshot ordering cannot misalign a cached forecast.
        cached = self._cache.get(key)
        by_id = (cached if isinstance(cached, dict)
                 and all(mid in cached for mid in member_ids) else None)

        if by_id is None:
            probs = None
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.llm_model,
                    messages=[
                        {"role": "system", "content": JOINT_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=220 + 16 * len(members),  # room for the reasoning field
                )
                probs = _parse_joint(resp.choices[0].message.content or "",
                                     len(members))
            except Exception as e:
                logger.warning("joint forecast failed for %s: %s", label, e)
                probs = None

            if probs is None:
                # Fallback: independent forecasts, renormalized to sum to 1.
                # Not cached — we want to retry the real joint call next day.
                logger.info("family %s: independent+renormalize fallback", label)
                raw = [self.base.forecast(_question(m))[0].p_yes for m in members]
                total = sum(raw) or 1.0
                by_id = {m.market_id: r / total for m, r in zip(members, raw)}
            else:
                by_id = {m.market_id: p for m, p in zip(members, probs)}
                self._cache[key] = by_id
                _save_cache(self._cache)

        return {
            m.market_id: [Forecast(
                p_yes=_clamp(by_id[m.market_id]),
                source="joint",
                rationale=f"family '{label}' ({len(members)}-way exclusive)",
            )]
            for m in members
        }
