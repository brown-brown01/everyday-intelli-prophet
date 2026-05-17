# Setup & run instructions

## 1. Prerequisites

- Python 3.11+
- A **Prophet Arena API key** (`PA_SERVER_API_KEY`) — issued by the Prophet
  Arena operator.
- An **OpenRouter API key** (`OPENROUTER_API_KEY`) — from
  [openrouter.ai](https://openrouter.ai). Powers the LLM forecaster; `:online`
  models add web search (~$0.02/call).

## 2. Install

```bash
cd everyday-intelli-prophet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in the two keys:

- `PA_SERVER_API_KEY` — your Prophet Arena key
- `OPENROUTER_API_KEY` — your OpenRouter key

Every other value has a sensible default. Confirm `PA_N_TICKS` matches your
intended run length (`1344` ≈ 14 days at 96 ticks/day).

## 4. Verify before launch

```bash
pytest -q tests/                  # 78 unit tests — all should pass
python -m prophet_arena.smoke     # end-to-end smoke (30 markets, live snapshot)
python -m prophet_arena.smoke 256 # full-snapshot stress run
```

The smoke test runs a live snapshot through the entire pipeline **without**
creating an experiment or submitting any trades.

Offline forecast scoring (needs no Prophet Arena key):

```bash
prophet forecast retrieve --dataset sample-resolved --include-resolved \
    -o data/prophet/resolved.json
python -m prophet_arena.harness
```

## 5. Launch the live bot

The bot is a continuous **~14-day process** — run it somewhere that stays up,
not a terminal you will close:

```bash
tmux new -s prophet
cd everyday-intelli-prophet
source .venv/bin/activate
python -m prophet_arena.bot 2>&1 | tee -a data/prophet/bot.log
```

Detach with **Ctrl-b** then **d**; reattach with `tmux attach -t prophet`. If
your machine sleeps or drops network, run it on a small always-on cloud VM
instead.

On launch the bot creates the experiment, then claims a tick every ~15 minutes:
forecast all markets → temper → decide → submit up to 20 trades → finalize. The
first tick of each day is the slow one (forecasts refresh, ~2–3 min); the rest
reuse the daily cache.

## 6. Resume

If the bot stops or crashes, **re-run the exact same command**. The slug returns
the existing experiment, and `data/prophet/` (controller state, forecast cache,
forecast log) persists on disk — the bot resumes where it left off.

## 7. Monitor

```bash
tail -f data/prophet/bot.log
prophet trade dashboard --slug everyday-intelli-prophet
```

The adaptive controller's `llm_weight` change log lives in
`data/prophet/controller_state.json`.

## Operational rules

1. **One process per slug.** Never run two bot instances against the same
   `PA_SLUG` — they fight over the tick lease and both lose.
2. **Keep the OpenRouter account funded** (~$35–45 for a 14-day run). If credits
   run out the forecaster falls back to a heuristic and the bot goes dormant —
   no crash, but no real trades.
