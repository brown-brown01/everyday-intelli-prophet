# everyday-intelli-prophet

A custom trading agent for the **Prophet Arena** prediction-market benchmark. It
forecasts every candidate market with a research-grounded LLM, tempers that
forecast against the market price with a Bayesian opinion pool, sizes trades
with fractional Kelly, and adapts itself online.

## What it is

Prophet Arena is a paper-trading benchmark: the server pins markets and prices
into 15-minute "ticks", runs deterministic fills, and scores participants on
mark-to-market P&L. This agent is a custom `BenchmarkSession` client — it
replaces the platform's built-in pipeline with its own forecasting + trading
brain.

## Pipeline (per 15-minute tick)

```
Prophet Arena snapshot  →  ~256 markets with live prices
   │
   ├─ families      group markets by mutual-exclusivity; one joint LLM
   │                call per family — market-anchored and web-searched
   │
   ├─ bayes.temper  blend the LLM forecast with the market mid-price as
   │                the prior, in log-odds space
   │
   ├─ pomdp         a per-market regime estimate modulates trust
   │
   ├─ strategy      EV vs price; if edge clears the threshold, size with
   │                fractional Kelly (shrunk for uncertainty + drawdown)
   │
   ├─ edge-rank     submit the top-N trades by edge
   │
   └─ controller    daily, tune llm_weight toward how well forecasts
                    predicted the market's own price drift
```

## Modules

| File | Role |
|---|---|
| `bot.py` | Prophet Arena tick loop — claim, forecast, decide, submit, finalize |
| `config.py` | Env-driven configuration |
| `forecaster.py` | LLM forecaster (any OpenAI-compatible endpoint) + no-key heuristic fallback |
| `families.py` | Family-aware **joint** forecasting — market-anchored, coherent |
| `bayes.py` | Bayesian tempering — log-odds opinion pool on `BayesianUpdater` |
| `bayesian_core.py` | Sequential Bayesian updater (log-odds math) |
| `strategy.py` | EV check + fractional-Kelly sizing + drawdown shrink |
| `controller.py` | Price-convergence controller — adapts `llm_weight` online |
| `pomdp.py` | Per-market regime (stable / volatile) belief |
| `calibration.py` | Post-hoc forecast calibration |
| `information.py` | Ensemble diversity / redundant-sample pruning |
| `harness.py` | Offline scorer against a resolved sample dataset |
| `model_eval.py` | Multi-model forecasting backtest |
| `smoke.py` | End-to-end smoke test |

## How it works

**Market-anchored joint forecasting.** Mutually-exclusive markets (e.g. all
"2028 Democratic nominee" candidates) share a `family` id. Forecasting them
independently yields probabilities that don't sum to 1 and spray false "edge".
`families.py` forecasts each family in one LLM call that **sees the current
market price for every candidate** and is told to start from it and deviate
only with concrete evidence — so the forecast is coherent and conservative.

**Bayesian tempering.** `bayes.temper()` treats the market mid-price and the
LLM forecast as two opinions and pools them in log-odds space, built on the
`BayesianUpdater` core. `LLM_WEIGHT` controls how far the posterior moves off
the market.

**Research.** With `LLM_MODEL=openai/gpt-4o:online`, every forecast is
web-search-grounded — the model justifies its deviations with current
information rather than stale training knowledge.

**Trading.** `strategy.decide()` trades a market only when expected value per
share clears `EV_THRESHOLD`, sizes with fractional Kelly (`KELLY_FRACTION`),
and shrinks for forecast uncertainty and portfolio drawdown. Candidate trades
are ranked by edge; the top `MAX_TRADES_PER_TICK` are submitted, all within the
$1k/market and $10k gross exposure caps.

**Adaptive controller.** Markets rarely resolve inside a 2-week window, so
resolution outcomes give no usable signal. Instead the controller measures, in
log-odds, whether the raw LLM forecast predicted the market's own subsequent
price drift (`beta = OLS(drift ~ disagreement)`) and slowly moves `LLM_WEIGHT`
toward that coefficient — a bounded, fail-safe daily update.

## Configuration

Set in `.env` (copy from `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `PA_SERVER_API_KEY` | — | Prophet Arena key (required to trade live) |
| `PA_SLUG` | `everyday-intelli-prophet` | Experiment slug — one bot, one slug |
| `PA_N_TICKS` | `1344` | Tick budget (~14 days × 96/day) |
| `OPENROUTER_API_KEY` | — | LLM key (required for real forecasts) |
| `LLM_MODEL` | `openai/gpt-4o:online` | Forecasting model; `:online` = web search |
| `LLM_WEIGHT` | `0.7` | Opinion-pool weight on the LLM vs the market |
| `EV_THRESHOLD` | `0.05` | Minimum edge per share to trade |
| `KELLY_FRACTION` | `0.25` | Fraction of full Kelly to stake |

## Running

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for full setup. In short:

```bash
python -m prophet_arena.harness   # offline forecast scorer (no PA key needed)
python -m prophet_arena.smoke     # end-to-end smoke test
python -m prophet_arena.bot       # live tick loop (needs PA + LLM keys)
```

## Testing

```bash
pytest -q tests/
```

78 tests cover the Bayesian tempering math, the adaptive controller (β
estimation, update rule, bounds, freeze), the trade-decision risk caps, and the
family-coherence invariants. `smoke.py` validates the full pipeline end-to-end
on a live market snapshot.

## Design notes

- **Long-dated universe.** The Prophet Arena market universe is dominated by
  markets resolving far in the future (median ~1.6 years). Over a 2-week eval
  window almost nothing resolves, so the agent is scored on **mark-to-market**
  P&L — it profits when prices drift toward its positions. The adaptive
  controller is built around this fact: it learns from price drift, not
  resolution outcomes.
- **Cost.** With `gpt-4o:online`, forecasting is cached per family per day —
  roughly $35–45 in LLM spend for a full 14-day run.
- **`shares` semantics.** Prophet Arena's `TradeIntentRequest.shares` is a
  **share count** (`notional = shares × price`), verified against the live API.
