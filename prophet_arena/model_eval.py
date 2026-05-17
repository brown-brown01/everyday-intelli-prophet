"""Fast multi-model forecasting backtest.

Scores several LLMs on resolved questions WITHOUT waiting for live
trading to play out. Each question is forecast *as of the day before it
closed*, so the model genuinely forecasts instead of noticing the event
already resolved and refusing. Reports Brier score, accuracy and log
loss per model so you can pick the forecaster brain before launch.

Calls run in parallel across (model, event) pairs and are disk-cached,
so the whole comparison takes a couple of minutes and re-runs for free.

    python -m prophet_arena.model_eval [events.json]
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timedelta

from .config import DATA_DIR, load_config
from .forecaster import _parse

# Prophet Arena model id  ->  OpenRouter model id
MODELS: dict[str, str] = {
    "openai:gpt-4o-mini":        "openai/gpt-4o-mini",
    "openai:gpt-4o":             "openai/gpt-4o",
    "anthropic:claude-sonnet-4": "anthropic/claude-sonnet-4",
    "gemini:gemini-2.5-flash":   "google/gemini-2.5-flash",
    "openai:gpt-5.2":            "openai/gpt-5.2",
}
# rough OpenRouter $/1M tokens (input, output)
PRICE: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.5, 10.0),
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "google/gemini-2.5-flash": (0.30, 2.5),
    "openai/gpt-5.2": (1.25, 10.0),
}

_CACHE_PATH = DATA_DIR / "model_eval_cache.json"

SYSTEM = (
    "You are a calibrated superforecaster. You are forecasting a future "
    "event and must NOT use any knowledge from after the stated 'as of' "
    "date. Weigh base rates and the information plausibly available as of "
    "that date. Avoid overconfidence. Respond with ONLY a JSON object: "
    '{"probability": <number 0-1>, "reasoning": "<one sentence>"}.'
)


def _yes_outcome(event: dict) -> str:
    outs = event.get("outcomes") or []
    return outs[0] if outs else "Yes"


def _resolved_yes(event: dict):
    ro = event.get("resolved_outcome")
    if not ro:
        return None
    return _yes_outcome(event) in (ro.get("value") or [])


def _as_of(close_time: str) -> str:
    """Day before the market closed — the forecast vantage point."""
    try:
        dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        return (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return "the day before it closed"


def _prompt(event: dict) -> str:
    outs = event.get("outcomes") or []
    target = _yes_outcome(event)
    lines = [
        f"You are forecasting AS OF {_as_of(event.get('close_time', ''))}.",
        f"Question: {event.get('title', '')}",
        f"Resolution rules: {event.get('rules') or event.get('description') or ''}",
    ]
    if len(outs) >= 2 and target.lower() not in ("yes", "no"):
        lines.append(
            f"This binary market resolves YES if and only if '{target}' is "
            f"the winning outcome. Candidate set: {', '.join(outs)}."
        )
    lines.append('Give P(YES) as JSON: {"probability": ..., "reasoning": ...}')
    return "\n".join(lines)


def forecast_one(client, model_or_id: str, prompt: str, cache: dict) -> dict:
    """One forecast. Always returns dict(p, pt, ct, lat, key, err)."""
    key = hashlib.sha256(f"{model_or_id}|{SYSTEM}|{prompt}".encode()).hexdigest()
    if key in cache:
        c = cache[key]
        return dict(p=c["p"], pt=c.get("pt", 0), ct=c.get("ct", 0),
                    lat=0.0, key=key, err=None)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_or_id,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,  # headroom for reasoning models
        )
        p, _ = _parse(resp.choices[0].message.content or "")
        u = resp.usage
        return dict(p=p, pt=u.prompt_tokens, ct=u.completion_tokens,
                    lat=time.time() - t0, key=key, err=None)
    except Exception as e:
        return dict(p=None, pt=0, ct=0, lat=time.time() - t0, key=key,
                    err=str(e))


def run(path: str | None = None, research: bool = False) -> dict:
    cfg = load_config()
    if not cfg.llm_api_key:
        raise SystemExit("No LLM key set (OPENROUTER_API_KEY). Cannot evaluate models.")
    from openai import OpenAI

    path = path or str(DATA_DIR / "resolved.json")
    events = json.loads(open(path).read())
    resolved = [e for e in events if _resolved_yes(e) is not None]

    try:
        cache = json.loads(_CACHE_PATH.read_text())
    except Exception:
        cache = {}

    # Research mode: evaluate the top models both plain and with OpenRouter
    # ':online' web search, so the no-search vs with-search rows sit side by
    # side. Restricted to 3 models to keep the web-plugin cost contained.
    if research:
        picks = ("openai:gpt-4o", "openai:gpt-4o-mini", "gemini:gemini-2.5-flash")
        model_set: list[tuple[str, str]] = []
        for pa in picks:
            model_set.append((pa, MODELS[pa]))
            model_set.append((pa + " +research", MODELS[pa] + ":online"))
    else:
        model_set = list(MODELS.items())

    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, timeout=120)
    prompts = [(_prompt(e), bool(_resolved_yes(e))) for e in resolved]

    print(f"\nMulti-model forecasting backtest — {len(resolved)} resolved events, "
          f"{len(model_set)} model runs, as-of dated"
          f"{'  [research = :online web search]' if research else ''}")
    print(f"({len(resolved) * len(model_set)} forecasts total; cached results reused)\n")

    tasks = [(pa, orid, pr, y)
             for pa, orid in model_set
             for (pr, y) in prompts]

    def _do(task):
        pa, orid, pr, y = task
        res = forecast_one(client, orid, pr, cache)
        return (pa, orid, y, res)

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_do, tasks))
    client.close()

    # collect per-model + write cache
    stats: dict[str, dict] = {pa: {"p": [], "y": [], "pt": 0, "ct": 0,
                                   "lat": [], "fail": 0, "orid": orid}
                              for pa, orid in model_set}
    for pa, orid, y, res in results:
        if res["p"] is None:
            stats[pa]["fail"] += 1
            continue
        stats[pa]["p"].append(res["p"])
        stats[pa]["y"].append(int(y))
        stats[pa]["pt"] += res["pt"]
        stats[pa]["ct"] += res["ct"]
        if res["lat"] > 0:
            stats[pa]["lat"].append(res["lat"])
        cache[res["key"]] = {"p": res["p"], "pt": res["pt"], "ct": res["ct"]}

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache))
    except Exception:
        pass

    # ── report ──
    rows = []
    for pa, s in stats.items():
        n = len(s["p"])
        if n == 0:
            rows.append((pa, None))
            continue
        brier = sum((p - y) ** 2 for p, y in zip(s["p"], s["y"])) / n
        acc = sum((p > 0.5) == bool(y) for p, y in zip(s["p"], s["y"])) / n
        ll = sum(-(y * math.log(min(max(p, 1e-6), 1 - 1e-6))
                   + (1 - y) * math.log(min(max(1 - p, 1e-6), 1)))
                 for p, y in zip(s["p"], s["y"])) / n
        pin, pout = PRICE.get(s["orid"].replace(":online", ""), (0, 0))
        cost = (s["pt"] * pin + s["ct"] * pout) / 1e6
        if ":online" in s["orid"]:
            cost += 0.02 * n  # OpenRouter web-search plugin, ~$0.02/call
        lat = sum(s["lat"]) / len(s["lat"]) if s["lat"] else 0.0
        rows.append((pa, dict(n=n, brier=brier, acc=acc, ll=ll, cost=cost,
                              lat=lat, fail=s["fail"])))

    rows.sort(key=lambda r: r[1]["brier"] if r[1] else 9.9)
    print(f"{'model':<28}{'Brier':>9}{'Acc':>8}{'LogLoss':>10}{'avg lat':>10}{'$/run':>9}")
    print("-" * 74)
    for pa, m in rows:
        if m is None:
            print(f"{pa:<28}{'— all calls failed —':>46}")
            continue
        flag = f"  ({m['fail']} failed)" if m["fail"] else ""
        print(f"{pa:<28}{m['brier']:>9.4f}{m['acc'] * 100:>7.1f}%"
              f"{m['ll']:>10.4f}{m['lat']:>9.1f}s{m['cost']:>9.4f}{flag}")
    best = next((r for r in rows if r[1]), None)
    print(f"\nlower Brier = better calibrated.  ran in {time.time() - t0:.0f}s")
    if best:
        print(f"best Brier: {best[0]}  ({best[1]['brier']:.4f})")
    return {pa: m for pa, m in rows}


if __name__ == "__main__":
    research = "research" in sys.argv
    path = next((a for a in sys.argv[1:] if a != "research"), None)
    run(path, research=research)
