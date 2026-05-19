"""Local web dashboard for the Prophet Arena bot.

Starts an HTTP server (port 8501 by default), opens a browser, and serves
a live monitoring interface for the running experiment.  All API calls are
proxied through this Python process so the API key never reaches the browser.
Polls every 10 seconds.

    python -m prophet_arena.dashboard
    python -m prophet_arena.dashboard --slug my-bot-v1
    python -m prophet_arena.dashboard --slug my-bot-v1 --port 9000

Reads PA_SERVER_URL, PA_SERVER_API_KEY, PA_SLUG, PA_REPORTING_API_URL from
the project .env file automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_DEFAULT_API_URL = os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev")
_DEFAULT_REPORTING_URL = (
    os.environ.get("PA_REPORTING_API_URL")
    or "https://trade-ui-api-998105805337.us-central1.run.app"
)

# Module-level state shared between the HTTP handler and open_dashboard().
_state: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path, query = parsed.path, parsed.query
        if path.startswith("/api/"):
            self._proxy(path[4:], query, base=_state["api_url"],
                        key=_state["api_key"], scope=True)
        elif path.startswith("/report/"):
            # Strip "/report" (7 chars) → keeps leading slash for the remote URL.
            self._proxy(path[7:], query, base=_state["reporting_url"],
                        key="", scope=False)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_state["html"])

    def _proxy(self, path: str, query: str, *, base: str,
               key: str, scope: bool) -> None:
        url = f"{base}{path}"
        if query:
            url += f"?{query}"
        try:
            headers = {"X-API-Key": key} if key else None
            r = httpx.get(url, timeout=15, headers=headers)
            body = r.content
            slug = _state.get("slug", "")
            # Scope /experiments list to the configured slug so the dashboard
            # always resolves the correct experiment even on shared API keys.
            if scope and slug and path == "/experiments" and r.status_code == 200:
                items = r.json()
                if isinstance(items, list):
                    body = json.dumps(
                        [e for e in items if e.get("experiment_slug") == slug]
                    ).encode()
                elif isinstance(items, dict) and isinstance(
                    items.get("experiments"), list
                ):
                    body = json.dumps(
                        {
                            **items,
                            "experiments": [
                                e for e in items["experiments"]
                                if e.get("experiment_slug") == slug
                            ],
                        }
                    ).encode()
            self.send_response(r.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())

    def log_message(self, *_):  # silence per-request logs
        pass


def open_dashboard(
    *,
    api_url: str = _DEFAULT_API_URL,
    slug: str = "",
    api_key: str = "",
    reporting_url: str = _DEFAULT_REPORTING_URL,
    port: int = 8501,
    block: bool = True,
) -> None:
    """Start the local proxy server and open the dashboard in the browser.

    Args:
        api_url:       Prophet Arena core API base URL.
        slug:          Experiment slug to scope the view.
        api_key:       PA_SERVER_API_KEY — stays in this process, never sent
                       to the browser.
        reporting_url: Read-only reporting API (PnL history, leaderboard).
        port:          Local TCP port (default 8501).
        block:         If True, block until the user presses Ctrl-C.
    """
    _state["api_url"] = api_url.rstrip("/")
    _state["api_key"] = api_key
    _state["reporting_url"] = reporting_url.rstrip("/")
    _state["slug"] = slug
    _state["html"] = _HTML.replace("__SLUG__", json.dumps(slug)).encode()

    srv = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    webbrowser.open(f"http://localhost:{port}")
    print(f"  Dashboard → http://localhost:{port}")
    if slug:
        print(f"  Experiment: {slug}")
    print(f"  API:        {_state['api_url']}")

    if block:
        try:
            print("  Press Ctrl-C to stop.")
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        finally:
            srv.server_close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m prophet_arena.dashboard``."""
    p = argparse.ArgumentParser(
        prog="python -m prophet_arena.dashboard",
        description="Live dashboard for the Prophet Arena bot.",
    )
    p.add_argument(
        "--slug",
        default=os.environ.get("PA_SLUG", ""),
        help="Experiment slug (default: PA_SLUG from .env)",
    )
    p.add_argument(
        "--api-url",
        default=_DEFAULT_API_URL,
        help="Core API base URL (default: PA_SERVER_URL from .env)",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("PA_SERVER_API_KEY", ""),
        help="API key (default: PA_SERVER_API_KEY from .env)",
    )
    p.add_argument(
        "--reporting-url",
        default=_DEFAULT_REPORTING_URL,
        help="Read-only reporting API URL (default: PA_REPORTING_API_URL from .env)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local server port (default: 8501)",
    )
    args = p.parse_args(argv)

    if not args.api_key:
        p.error(
            "No API key found.  Set PA_SERVER_API_KEY in .env or pass --api-key."
        )

    open_dashboard(
        api_url=args.api_url,
        slug=args.slug,
        api_key=args.api_key,
        reporting_url=args.reporting_url,
        port=args.port,
        block=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Self-contained HTML dashboard
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prophet Bot Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#09090b;--fg:#f4f4f5;--sub:#a1a1aa;--muted:#71717a;--border:#27272a;
  --green:#22c55e;--red:#ef4444;--blue:#60a5fa;
  --p0:#60a5fa;--p1:#4ade80;--p2:#fb923c;--p3:#c084fc
}
html,body{min-height:100%;background:var(--bg);color:var(--fg);
  font-family:'JetBrains Mono','SF Mono',ui-monospace,monospace;font-size:13px;line-height:1.45}
main{max-width:1400px;margin:0 auto;padding:36px 24px 60px}

/* Header */
.hdr{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;
  margin-bottom:22px;padding-bottom:20px;border-bottom:1px solid var(--border)}
.hdr h1{font-size:22px;font-weight:700;color:#fafafa}
.hdr-meta{font-size:11px;color:var(--muted);text-align:right;white-space:nowrap}

/* Panels */
.panel{background:rgba(17,17,19,.76);border:1px solid var(--border);
  border-radius:6px;overflow:hidden;margin-bottom:22px}
.panel-hd{display:flex;align-items:baseline;justify-content:space-between;
  gap:12px;padding:14px 16px 10px;border-bottom:1px solid var(--border)}
.panel-hd h2{font-size:13px;font-weight:500;color:#e4e4e7}
.panel-hd span{font-size:11px;color:var(--muted)}

/* KV table */
.kv{margin:0;width:100%;font-size:13px;line-height:1.5}
.kv td{padding:11px 18px;border-bottom:1px solid var(--border);vertical-align:top}
.kv td:first-child{color:var(--sub);width:180px}
.kv td:last-child{color:var(--fg);text-align:right;font-variant-numeric:tabular-nums}
.kv .v{font-weight:600;white-space:nowrap}
.kv .s{color:var(--muted);font-size:11px;font-weight:400;margin-top:3px}
.kv tr:last-child td{border-bottom:none}

/* Chart */
.chart{height:300px;padding:12px 14px 16px;position:relative}
.chart svg{display:block;width:100%;height:100%}
.chart svg circle{cursor:pointer}
.chart svg circle:hover{stroke-width:2.5;r:5}
.tt{position:absolute;transform:translate(-50%,calc(-100% - 10px));background:#18181b;
  border:1px solid var(--border);border-radius:6px;padding:8px 11px;font-size:11px;
  pointer-events:none;white-space:nowrap;z-index:10;
  box-shadow:0 6px 18px rgba(0,0,0,.55);min-width:170px}
.tt-row{display:flex;justify-content:space-between;gap:18px;line-height:1.6}
.tt-row .k{color:var(--muted)}.tt-row .v{color:var(--fg);font-weight:600}

/* Chart legend */
.legend{display:flex;gap:16px;padding:4px 14px 2px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--sub)}
.legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}

/* Tables */
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#141416;color:var(--muted);font-size:10px;font-weight:500;
  text-transform:uppercase;letter-spacing:.11em;text-align:left;
  padding:9px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th.r,td.r{text-align:right}
tbody tr:hover{background:rgba(39,39,42,.45)}
.clip{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:340px}
.scroll{max-height:420px;overflow:auto}
.empty{color:var(--muted);padding:32px 16px;text-align:center;font-size:12px}
.warn{border:1px solid rgba(245,158,11,.35);background:rgba(245,158,11,.08);
  color:#fcd34d;border-radius:5px;padding:9px 11px;font-size:11px;margin-bottom:14px}

/* Badges */
.badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;
  font-weight:600;letter-spacing:.07em;text-transform:uppercase}
.b-open{background:rgba(96,165,250,.15);color:#60a5fa}
.b-won{background:rgba(34,197,94,.15);color:#22c55e}
.b-lost{background:rgba(239,68,68,.15);color:#ef4444}
.b-closed{background:rgba(161,161,170,.15);color:#a1a1aa}

/* Ledger tabs */
.tabs{display:flex;gap:8px;padding:12px 16px 0}
.tab{padding:6px 14px;border-radius:4px 4px 0 0;cursor:pointer;font-size:12px;
  color:var(--muted);border:1px solid transparent;border-bottom:none;
  background:rgba(0,0,0,.2)}
.tab.active{color:var(--fg);background:rgba(39,39,42,.8);border-color:var(--border)}
.tab:hover:not(.active){color:var(--sub)}

/* Search */
.search-row{display:flex;gap:10px;align-items:center;padding:12px 16px 8px}
.search-row input{flex:1;background:#141416;border:1px solid var(--border);
  border-radius:4px;padding:7px 11px;color:var(--fg);font-family:inherit;
  font-size:12px;outline:none}
.search-row input:focus{border-color:#52525b}
.search-row input::placeholder{color:var(--muted)}

/* Reasoning cards */
.reason-wrap{padding:12px 16px 16px}
.reason-btn{background:rgba(39,39,42,.6);border:1px solid var(--border);
  border-radius:4px;color:var(--sub);padding:7px 16px;cursor:pointer;
  font-family:inherit;font-size:12px;margin-bottom:14px}
.reason-btn:hover{color:var(--fg);border-color:#52525b}
.reason-btn:disabled{opacity:.5;cursor:default}
.rcard{border:1px solid var(--border);border-radius:5px;margin-bottom:10px;overflow:hidden}
.rcard-hd{display:flex;justify-content:space-between;align-items:center;
  padding:10px 14px;cursor:pointer;background:rgba(20,20,22,.9)}
.rcard-hd:hover{background:rgba(39,39,42,.7)}
.rcard-hd .tick-info{font-size:12px;color:#e4e4e7}
.rcard-hd .tick-sub{font-size:11px;color:var(--muted);margin-left:12px}
.rcard-hd .chev{font-size:10px;color:var(--muted)}
.rcard-body{padding:12px 14px;font-size:11px;color:var(--sub)}
.rcard-body.hide{display:none}
.rtrade{padding:6px 0;border-bottom:1px solid var(--border);
  display:grid;grid-template-columns:1fr auto auto auto;gap:10px;align-items:start}
.rtrade:last-child{border-bottom:none}
.rtrade .q{line-height:1.4}
.rtrade .q .mid{font-size:10px;color:var(--muted);margin-top:2px}

/* Utilities */
.mono{font-variant-numeric:tabular-nums}
.green{color:var(--green)!important}.red{color:var(--red)!important}
.blue{color:var(--blue)!important}.muted{color:var(--muted)!important}
.hide{display:none!important}

@media(max-width:720px){
  main{padding:22px 14px 40px}.hdr{flex-direction:column;align-items:flex-start}
  .hdr-meta{text-align:left}.chart{height:220px}.clip{max-width:200px}
}
</style>
</head>
<body>
<main>
  <div class="hdr">
    <h1 id="runSlug">Prophet Bot</h1>
    <div class="hdr-meta mono">
      <div id="runStatus">connecting…</div>
      <div id="runStarted" class="muted"></div>
      <div id="lastPoll" class="muted"></div>
    </div>
  </div>

  <div id="err" class="hide warn"></div>

  <!-- Metrics -->
  <section class="panel">
    <div class="panel-hd"><h2>Metrics</h2><span id="metaHint"></span></div>
    <table class="kv"><tbody id="metrics"></tbody></table>
  </section>

  <!-- Equity chart -->
  <section class="panel">
    <div class="panel-hd"><h2>Equity over ticks</h2><span id="chartMeta">waiting for data</span></div>
    <div id="legend" class="legend hide"></div>
    <div class="chart" id="chart">
      <div id="chartTT" class="tt hide"></div>
    </div>
  </section>

  <!-- Leaderboard -->
  <section class="panel">
    <div class="panel-hd"><h2>Leaderboard</h2><span id="boardMeta"></span></div>
    <div id="leaderboard"></div>
  </section>

  <!-- Market ledger -->
  <section class="panel">
    <div class="panel-hd"><h2>Market Ledger</h2><span id="ledgerMeta"></span></div>
    <div class="tabs" id="ledgerTabs">
      <div class="tab active" data-tab="open">Open</div>
      <div class="tab" data-tab="trades">All Trades</div>
      <div class="tab" data-tab="won">Won</div>
      <div class="tab" data-tab="lost">Lost</div>
    </div>
    <div class="search-row">
      <input id="ledgerSearch" type="text" placeholder="Search market ID or question…">
    </div>
    <div id="ledger"></div>
  </section>

  <!-- Reasoning -->
  <section class="panel">
    <div class="panel-hd"><h2>Reasoning</h2><span id="reasonMeta">lazy-loaded from /experiments/{id}/reasoning</span></div>
    <div class="reason-wrap">
      <button class="reason-btn" id="reasonBtn" onclick="loadReasoning()">Load last 5 ticks</button>
      <div id="reasoning"></div>
    </div>
  </section>
</main>

<script>
const API = '/api', REPORT = '/report';
const SLUG = __SLUG__;
const POLL_MS = 10000;
const MIN_DAILY_OBS = 3, MIN_WIN_RATE_TRADES = 10;
const P_COLORS = ['#60a5fa', '#4ade80', '#fb923c', '#c084fc'];

let exp = null, participants = [], portfolios = {}, pnlData = {}, fills = [];
let reasoningEntries = null;
let ledgerTab = 'open', ledgerQuery = '';

// ── Utilities ────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
const num = v => { const n = typeof v === 'string' ? Number(v) : v; return Number.isFinite(n) ? n : null; };
const fmt = (v, d = 2) => { const n = num(v); return n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }); };
const fmtI = v => { const n = num(v); return n == null ? '—' : Math.round(n).toLocaleString(); };
const usd = v => { const n = num(v); return n == null ? '—' : n.toLocaleString(undefined, { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }); };
const susd = v => { const n = num(v); return n == null ? '—' : `${n >= 0 ? '+' : '-'}${usd(Math.abs(n))}`; };
const pct = (v, d = 2) => { const n = num(v); return n == null ? '—' : `${n >= 0 ? '+' : ''}${(n * 100).toFixed(d)}%`; };
const fmtTime = iso => { if (!iso) return '—'; const d = new Date(iso); return isFinite(d) ? d.toLocaleString() : '—'; };

function elapsed(iso) {
  if (!iso) return '';
  const t = new Date(iso); if (!isFinite(t)) return '';
  const s = Math.max(0, (Date.now() - t.getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h ago`;
  return `${(s / 86400).toFixed(1)}d ago`;
}

function asArr(p, keys) {
  if (Array.isArray(p)) return p;
  for (const k of keys) if (Array.isArray(p?.[k])) return p[k];
  return [];
}

// ── Data loading ─────────────────────────────────────────────────────────────

async function get(path, opts = {}) {
  const r = await fetch(API + path);
  if (r.ok) return r.json();
  if (!opts.optional) throw new Error(`${path}: HTTP ${r.status}`);
  return null;
}

async function reportGet(path, opts = {}) {
  const r = await fetch(REPORT + path);
  if (r.ok) return r.json();
  if (!opts.optional) throw new Error(`${path}: HTTP ${r.status}`);
  return null;
}

function rowTs(r) { return r?.tick_ts || r?.timestamp || r?.created_at || r?.updated_at || r?.as_of || r?.date; }
function rowEq(r) { return num(r?.equity ?? r?.portfolio_value ?? r?.total_equity); }

async function load() {
  const list = await get('/experiments', { optional: true });
  let exps = asArr(list, ['experiments']);
  exps.sort((a, b) =>
    (a.status === 'RUNNING' ? 0 : 1) - (b.status === 'RUNNING' ? 0 : 1) ||
    (b.last_activity_at || '').localeCompare(a.last_activity_at || '')
  );
  if (!exps.length) { exp = null; return; }
  const id = exps[0].experiment_id;

  const [detail, prog, partsResp, fillsResp] = await Promise.all([
    get('/experiments/' + id, { optional: true }),
    get('/experiments/' + id + '/progress', { optional: true }),
    get('/experiments/' + id + '/participants', { optional: true }),
    get('/experiments/' + id + '/trades?limit=500', { optional: true }),
  ]);

  exp = detail || exps[0];
  if (prog) { exp._done = prog.completed || 0; exp._total = prog.n_ticks || exp.n_ticks; }
  else { exp._done = exp.completed_ticks ?? exp.completed; exp._total = exp.n_ticks; }

  participants = asArr(partsResp, ['participants']);
  if (!participants.length) participants = [{ participant_idx: 0, model: 'custom:bot' }];

  fills = asArr(fillsResp, ['fills', 'trades']).filter(
    f => Number(f.participant_idx ?? 0) === 0
  );
  fills.sort((a, b) => new Date(b.filled_at || b.timestamp) - new Date(a.filled_at || a.timestamp));

  // Portfolio + PnL per participant (cap at 4)
  await Promise.all(participants.slice(0, 4).map(async p => {
    const idx = p.participant_idx ?? p.idx ?? 0;
    const [port, pnlResp] = await Promise.all([
      get('/experiments/' + id + '/participants/' + idx + '/portfolio', { optional: true }),
      reportGet('/experiments/' + id + '/pnl?participant_idx=' + idx, { optional: true }),
    ]);
    portfolios[idx] = port;
    let pts = asArr(pnlResp, ['pnl']).filter(r => Number(r.participant_idx ?? 0) === idx);
    pts.sort((a, b) => new Date(rowTs(a)) - new Date(rowTs(b)));
    pnlData[idx] = pts;
  }));
}

// ── Metric computation ────────────────────────────────────────────────────────

function tickPts(idx = 0) {
  return (pnlData[idx] || []).map(r => {
    const t = new Date(rowTs(r)); const e = rowEq(r);
    if (!isFinite(t) || e == null) return null;
    return { t, equity: e, cash: num(r.cash), totalPnl: num(r.total_pnl ?? r.totalPnl), positions: num(r.num_positions ?? r.numPositions) };
  }).filter(Boolean);
}

function dailyPts(idx = 0) {
  const map = new Map();
  for (const p of tickPts(idx)) {
    const k = p.t.toISOString().slice(0, 10);
    const prev = map.get(k);
    if (!prev || p.t > prev.t) map.set(k, p);
  }
  return [...map.values()].sort((a, b) => a.t - b.t);
}

function startCash(idx = 0) {
  const p = participants.find(x => (x.participant_idx ?? x.idx ?? 0) === idx);
  return num(p?.starting_cash ?? p?.startingCash) ?? 10000;
}

function computeMetrics(idx = 0) {
  const ticks = tickPts(idx), daily = dailyPts(idx);
  const start = startCash(idx);
  const port = portfolios[idx];
  const latest = ticks.length ? ticks[ticks.length - 1].equity : num(port?.equity ?? port?.total_equity);
  const totalPnl = latest != null ? latest - start : (num(port?.total_pnl) ?? null);
  const ret = latest != null && start ? (latest - start) / start : null;

  const eqs = daily.map(d => d.equity);
  const rets = [];
  for (let i = 1; i < eqs.length; i++) if (eqs[i - 1]) rets.push(eqs[i] / eqs[i - 1] - 1);

  const eligible = daily.length >= MIN_DAILY_OBS;
  const mean = rets.length ? rets.reduce((s, x) => s + x, 0) / rets.length : null;
  const variance = rets.length >= 2 && mean != null
    ? rets.reduce((s, x) => s + (x - mean) ** 2, 0) / (rets.length - 1) : null;
  const std = variance != null ? Math.sqrt(variance) : null;
  const sharpe = eligible && std != null ? (std === 0 ? 0 : (mean / std) * Math.sqrt(365)) : null;

  let cagr = null;
  if (daily.length >= 2 && daily[0].equity > 0) {
    const days = Math.max(0, (daily[daily.length - 1].t - daily[0].t) / 86400000);
    if (days > 0) cagr = (daily[daily.length - 1].equity / daily[0].equity) ** (365 / days) - 1;
  }

  let peak = eqs[0] ?? null, maxDd = null;
  if (eqs.length >= 2 && peak != null) {
    maxDd = 0;
    for (const e of eqs) { if (e > peak) peak = e; if (peak > 0) maxDd = Math.max(maxDd, (peak - e) / peak); }
  }

  return {
    start, equity: latest, totalPnl, returnPct: ret,
    cash: num(port?.cash), positions: port?.positions?.length ?? 0,
    sharpe: eligible ? sharpe : null, maxDd: eligible ? maxDd : null,
    cagr: eligible ? cagr : null, nObs: daily.length, eligible, ticks, daily,
  };
}

function winRate() {
  const groups = new Map(); let hasOutcome = false;
  for (const f of fills) {
    const outcome = f.market_outcome ?? f.outcome ?? null;
    if (outcome == null) continue;
    hasOutcome = true;
    const side = String(f.side || '').toUpperCase();
    const action = String(f.action || '').toUpperCase();
    const key = `${f.market_id}:${side}`;
    const g = groups.get(key) || { cost: 0, sell: 0, bought: 0, sold: 0, outcome, side };
    const shares = num(f.shares) ?? 0;
    const cost = Math.abs(num(f.notional ?? f.cost) || (shares * (num(f.price) ?? 0)));
    if (action === 'BUY') { g.cost += cost; g.bought += shares; }
    if (action === 'SELL') { g.sell += cost; g.sold += shares; }
    g.outcome = outcome;
    groups.set(key, g);
  }
  if (!hasOutcome) return { nTrades: 0, rate: null };
  let wins = 0, total = 0;
  for (const g of groups.values()) {
    const out = String(g.outcome).toUpperCase();
    const yes = out === 'YES' || out === '1' || out === 'TRUE' || g.outcome === 1;
    const payout = g.side === 'YES' ? (yes ? 1 : 0) : (yes ? 0 : 1);
    const pnl = g.sell + (g.bought - g.sold) * payout - g.cost;
    total++; if (pnl > 0) wins++;
  }
  return { nTrades: total, rate: total >= MIN_WIN_RATE_TRADES ? wins / total : null };
}

// ── Render ───────────────────────────────────────────────────────────────────

function render() {
  if (!exp) {
    $('metrics').innerHTML = '<tr><td colspan="2" class="empty">No experiment found.</td></tr>';
    $('leaderboard').innerHTML = '';
    $('ledger').innerHTML = '';
    return;
  }
  const m = computeMetrics(0), wr = winRate();
  const sc = exp.status === 'RUNNING' ? 'green' : exp.status === 'COMPLETED' ? 'blue' : exp.status === 'ABORTED' ? 'red' : '';
  $('runSlug').textContent = exp.experiment_slug || SLUG || 'Prophet Bot';
  $('runStatus').innerHTML = `<span class="${sc}">${esc(exp.status || '—')}</span> · ${exp._done ?? 0}/${exp._total ?? '—'} ticks`;
  const st = exp.started_at || exp.created_at;
  $('runStarted').textContent = st ? `started ${elapsed(st)}` : '';
  $('lastPoll').textContent = `updated just now`;
  renderMetrics(m, wr);
  renderChart();
  renderLeaderboard();
  renderLedger();
}

function kvRow(label, value, sub, cls = '') {
  return `<tr><td>${label}</td><td><div class="v ${cls}">${value}</div>${sub ? `<div class="s">${sub}</div>` : ''}</td></tr>`;
}
function gatedRow(label, value, gate) {
  return gate.pending
    ? kvRow(label, `<span class="muted">Pending</span>`, gate.sub)
    : kvRow(label, value, gate.sub);
}

function dailyGate(m) {
  if (m.eligible) return { pending: false, sub: `${m.nObs} daily P&L observations` };
  if (m.nObs === 0) return { pending: true, sub: `Needs ${MIN_DAILY_OBS} daily P&L obs. No UTC day boundary yet.` };
  if (m.nObs === 1) return { pending: true, sub: `1 of ${MIN_DAILY_OBS} days. Two more UTC midnights to go.` };
  return { pending: true, sub: `${m.nObs} of ${MIN_DAILY_OBS} days.` };
}
function tradesGate(wr) {
  if (wr.nTrades >= MIN_WIN_RATE_TRADES) return { pending: false, sub: `${wr.nTrades} resolved trades` };
  if (wr.nTrades === 0) return { pending: true, sub: `Needs ${MIN_WIN_RATE_TRADES} resolved markets. None settled yet.` };
  return { pending: true, sub: `${wr.nTrades} of ${MIN_WIN_RATE_TRADES} resolved.` };
}

function renderMetrics(m, wr) {
  const pnlCls = m.totalPnl == null ? '' : m.totalPnl >= 0 ? 'green' : 'red';
  const dg = dailyGate(m), tg = tradesGate(wr);
  $('metaHint').textContent = `participant 0 · ${fmtI(m.positions)} open positions`;
  $('metrics').innerHTML = [
    kvRow('Equity', usd(m.equity), `start ${usd(m.start)}`),
    kvRow('Total P&L', susd(m.totalPnl), pct(m.returnPct), pnlCls),
    kvRow('Cash', usd(m.cash), `${fmtI(m.positions)} open positions`),
    gatedRow('Sharpe', fmt(m.sharpe, 2), dg),
    gatedRow('Max Drawdown', pct(m.maxDd).replace('+', ''), dg),
    gatedRow('CAGR', pct(m.cagr), dg),
    gatedRow('Win Rate', pct(wr.rate, 1).replace('+', ''), tg),
  ].join('');
}

// ── Equity chart ─────────────────────────────────────────────────────────────

function renderChart() {
  const chartEl = $('chart'), ttEl = $('chartTT'), legendEl = $('legend');
  ttEl.classList.add('hide');

  let svgHost = chartEl.querySelector('.svg-host');
  if (!svgHost) {
    svgHost = document.createElement('div');
    svgHost.className = 'svg-host';
    svgHost.style.cssText = 'width:100%;height:100%';
    chartEl.insertBefore(svgHost, chartEl.firstChild);
  }

  const series = participants.slice(0, 4).map((p, ci) => {
    const idx = p.participant_idx ?? p.idx ?? 0;
    return { idx, model: p.model || `P${idx}`, color: P_COLORS[ci], pts: tickPts(idx) };
  }).filter(s => s.pts.length);

  if (!series.length) {
    $('chartMeta').textContent = 'P&L history unavailable';
    svgHost.innerHTML = '<div class="empty">P&L history unavailable.</div>';
    legendEl.classList.add('hide');
    return;
  }

  legendEl.innerHTML = series.map(s =>
    `<div class="legend-item"><div class="legend-dot" style="background:${s.color}"></div><span>${esc(s.model)}</span></div>`
  ).join('');
  legendEl.classList.toggle('hide', series.length === 1);

  const allEqs = series.flatMap(s => s.pts.map(p => p.equity));
  let minY = Math.min(...allEqs), maxY = Math.max(...allEqs);
  if (minY === maxY) { minY -= Math.max(1, minY * .02); maxY += Math.max(1, maxY * .02); }
  const padY = (maxY - minY) * .15; minY -= padY; maxY += padY;

  const W = 1000, H = 270, L = 64, R = 20, T = 12, B = 32;
  const maxN = Math.max(...series.map(s => s.pts.length));
  const xx = (i, n) => n === 1 ? L + (W - L - R) / 2 : L + (i / (n - 1)) * (W - L - R);
  const yy = v => T + (1 - (v - minY) / (maxY - minY)) * (H - T - B);

  const grid = [minY, (minY + maxY) / 2, maxY].map(v =>
    `<line x1="${L}" x2="${W - R}" y1="${yy(v).toFixed(1)}" y2="${yy(v).toFixed(1)}" stroke="#27272a"/>
     <text x="6" y="${(yy(v) + 4).toFixed(1)}" fill="#a1a1aa" font-size="10">${usd(v)}</text>`
  ).join('');

  const paths = series.map(s => {
    const n = s.pts.length;
    const d = s.pts.map((p, i) => `${i ? 'L' : 'M'}${xx(i, n).toFixed(1)},${yy(p.equity).toFixed(1)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
  }).join('');

  const primary = series[0], pn = primary.pts.length;
  const dots = primary.pts.map((p, i) =>
    `<circle data-i="${i}" cx="${xx(i, pn).toFixed(1)}" cy="${yy(p.equity).toFixed(1)}" r="4" fill="#0c0c0e" stroke="${primary.color}" stroke-width="1.6"/>`
  ).join('');

  const sameDay = pn >= 2 && primary.pts[0].t.toDateString() === primary.pts[pn - 1].t.toDateString();
  const fmtT = t => sameDay
    ? t.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : t.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + t.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

  const labelIs = [...new Set([0, Math.floor((maxN - 1) / 4), Math.floor((maxN - 1) / 2), Math.floor(3 * (maxN - 1) / 4), maxN - 1].map(i => Math.min(i, maxN - 1)))];
  const xLabels = labelIs.map(i => {
    const cx = xx(i, maxN).toFixed(1);
    const an = i === 0 ? 'start' : i === maxN - 1 ? 'end' : 'middle';
    const srcPt = primary.pts[Math.min(i, pn - 1)];
    return `<text x="${cx}" y="${H - 4}" fill="#71717a" font-size="10" text-anchor="${an}">${srcPt ? fmtT(srcPt.t) : '#' + (i + 1)}</text>`;
  }).join('');

  $('chartMeta').textContent = `${pn} tick${pn === 1 ? '' : 's'} · ${series.length} participant${series.length === 1 ? '' : 's'}`;

  svgHost.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${grid}
    <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#3f3f46"/>
    ${paths}${dots}${xLabels}
  </svg>`;

  svgHost.querySelectorAll('circle[data-i]').forEach(c => {
    c.addEventListener('mouseenter', () => {
      const i = parseInt(c.dataset.i), p = primary.pts[i];
      const rows = [['Tick', `#${i + 1}`], ['Time', fmtT(p.t)], ['Equity', usd(p.equity)]];
      if (p.cash != null) rows.push(['Cash', usd(p.cash)]);
      if (p.totalPnl != null) rows.push(['P&L', `<span class="${p.totalPnl >= 0 ? 'green' : 'red'}">${susd(p.totalPnl)}</span>`]);
      if (p.positions != null) rows.push(['Positions', fmtI(p.positions)]);
      ttEl.innerHTML = rows.map(([k, v]) => `<div class="tt-row"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
      const cr = c.getBoundingClientRect(), chr = chartEl.getBoundingClientRect();
      ttEl.style.left = (cr.left + cr.width / 2 - chr.left) + 'px';
      ttEl.style.top = (cr.top - chr.top) + 'px';
      ttEl.classList.remove('hide');
    });
    c.addEventListener('mouseleave', () => ttEl.classList.add('hide'));
  });
}

// ── Leaderboard ───────────────────────────────────────────────────────────────

function renderLeaderboard() {
  const rows = participants.slice(0, 8).map((p, ci) => {
    const idx = p.participant_idx ?? p.idx ?? 0;
    const port = portfolios[idx];
    const equity = num(port?.equity ?? port?.total_equity);
    const start = startCash(idx);
    const pnl = equity != null ? equity - start : num(port?.total_pnl) ?? null;
    const ret = equity != null && start ? (equity - start) / start : null;
    return { idx, model: p.model || `P${idx}`, rep: p.rep ?? 0, equity, pnl, ret, fills: port?.total_fills ?? 0, color: P_COLORS[ci] };
  }).sort((a, b) => (b.equity ?? -Infinity) - (a.equity ?? -Infinity));

  $('boardMeta').textContent = `${rows.length} participant${rows.length === 1 ? '' : 's'}`;
  if (!rows.length) { $('leaderboard').innerHTML = '<div class="empty">No participants.</div>'; return; }

  $('leaderboard').innerHTML = `<div class="scroll"><table>
    <thead><tr>
      <th>#</th><th>Model</th><th class="r">Equity</th>
      <th class="r">Total P&amp;L</th><th class="r">Return</th><th class="r">Fills</th>
    </tr></thead>
    <tbody>${rows.map((r, rank) => {
    const cls = r.pnl == null ? '' : r.pnl >= 0 ? 'green' : 'red';
    return `<tr>
        <td><span style="color:${r.color}">●</span> ${rank + 1}</td>
        <td><div class="clip" title="${esc(r.model)}">${esc(r.model)}</div></td>
        <td class="r mono">${usd(r.equity)}</td>
        <td class="r mono ${cls}">${susd(r.pnl)}</td>
        <td class="r mono ${cls}">${pct(r.ret)}</td>
        <td class="r mono">${fmtI(r.fills)}</td>
      </tr>`;
  }).join('')}</tbody>
  </table></div>`;
}

// ── Market Ledger ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  $('ledgerTabs').addEventListener('click', e => {
    const tab = e.target.dataset.tab; if (!tab) return;
    ledgerTab = tab;
    $('ledgerTabs').querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
    renderLedger();
  });
  $('ledgerSearch').addEventListener('input', e => {
    ledgerQuery = e.target.value.toLowerCase();
    renderLedger();
  });
});

function tradeStatus(f) {
  const outcome = f.market_outcome ?? f.outcome;
  if (outcome == null) return 'closed';
  const side = String(f.side || '').toUpperCase();
  const out = String(outcome).toUpperCase();
  const yes = out === 'YES' || out === '1' || out === 'TRUE' || outcome === 1;
  return (side === 'YES' && yes) || (side === 'NO' && !yes) ? 'won' : 'lost';
}

function renderLedger() {
  const port = portfolios[0];
  const positions = port?.positions || [];
  const openIds = new Set(positions.map(p => p.market_id));

  let items = [];
  if (ledgerTab === 'open') {
    items = positions.map(p => ({
      kind: 'pos', market_id: p.market_id, side: String(p.side || '').toUpperCase(),
      shares: p.shares, entry: p.avg_entry_price, mark: p.current_price,
      upnl: p.unrealized_pnl, status: 'open',
    }));
  } else {
    const filtered = fills.filter(f => {
      const st = tradeStatus(f);
      if (ledgerTab === 'won') return st === 'won';
      if (ledgerTab === 'lost') return st === 'lost';
      return true; // all trades
    });
    items = filtered.map(f => ({
      kind: 'fill', market_id: f.market_id,
      action: String(f.action || '').toUpperCase(),
      side: String(f.side || '').toUpperCase(),
      shares: f.shares, price: f.price,
      notional: f.notional ?? f.cost, ts: f.filled_at || f.timestamp,
      status: tradeStatus(f),
    }));
  }

  if (ledgerQuery) {
    items = items.filter(it =>
      it.market_id.toLowerCase().includes(ledgerQuery) ||
      (it.question || '').toLowerCase().includes(ledgerQuery)
    );
  }

  const label = ledgerTab === 'open' ? 'open positions' : ledgerTab === 'trades' ? 'trades' : ledgerTab + ' trades';
  $('ledgerMeta').textContent = `${items.length} ${label}`;

  if (!items.length) {
    $('ledger').innerHTML = `<div class="empty">${ledgerTab === 'won' || ledgerTab === 'lost' ? 'No resolved trades yet.' : 'No items.'}</div>`;
    return;
  }

  if (ledgerTab === 'open') {
    $('ledger').innerHTML = `<div class="scroll"><table>
      <thead><tr>
        <th>Market ID</th><th>Side</th>
        <th class="r">Shares</th><th class="r">Avg Entry</th>
        <th class="r">Mark</th><th class="r">Unrealized P&amp;L</th>
      </tr></thead>
      <tbody>${items.map(p => {
      const upnl = num(p.upnl);
      const sc = p.side === 'YES' ? 'green' : p.side === 'NO' ? 'red' : 'muted';
      return `<tr>
          <td><div class="clip">${esc(p.market_id)}</div></td>
          <td class="${sc}">${esc(p.side)}</td>
          <td class="r mono">${fmt(p.shares)}</td>
          <td class="r mono">${fmt(p.entry, 4)}</td>
          <td class="r mono">${fmt(p.mark, 4)}</td>
          <td class="r mono ${upnl == null ? '' : upnl >= 0 ? 'green' : 'red'}">${susd(upnl)}</td>
        </tr>`;
    }).join('')}</tbody>
    </table></div>`;
  } else {
    $('ledger').innerHTML = `<div class="scroll"><table>
      <thead><tr>
        <th>Time</th><th>Market ID</th><th>Action</th><th>Side</th>
        <th class="r">Shares</th><th class="r">Price</th><th class="r">Notional</th><th>Status</th>
      </tr></thead>
      <tbody>${items.map(it => {
      const ac = it.action === 'BUY' ? 'green' : it.action === 'SELL' ? 'red' : '';
      const bc = it.status === 'won' ? 'b-won' : it.status === 'lost' ? 'b-lost' : it.status === 'open' ? 'b-open' : 'b-closed';
      return `<tr>
          <td class="mono muted">${it.ts ? new Date(it.ts).toLocaleString() : '—'}</td>
          <td><div class="clip">${esc(it.market_id)}</div></td>
          <td class="${ac}">${esc(it.action || '—')}</td>
          <td>${esc(it.side)}</td>
          <td class="r mono">${fmt(it.shares)}</td>
          <td class="r mono">${fmt(it.price, 4)}</td>
          <td class="r mono">${usd(Math.abs(num(it.notional) || 0))}</td>
          <td><span class="badge ${bc}">${it.status}</span></td>
        </tr>`;
    }).join('')}</tbody>
    </table></div>`;
  }
}

// ── Reasoning (lazy-loaded) ───────────────────────────────────────────────────

async function loadReasoning() {
  if (!exp) return;
  const btn = $('reasonBtn');
  btn.textContent = 'Loading…'; btn.disabled = true;
  try {
    const resp = await get('/experiments/' + exp.experiment_id + '/reasoning?limit=5', { optional: true });
    reasoningEntries = asArr(resp, ['reasoning']);
    reasoningEntries.sort((a, b) => new Date(b.tick_id) - new Date(a.tick_id));
    $('reasonMeta').textContent = `${reasoningEntries.length} entr${reasoningEntries.length === 1 ? 'y' : 'ies'}`;
    renderReasoning();
  } catch (e) {
    $('reasoning').innerHTML = `<div class="empty">Failed to load: ${esc(String(e))}</div>`;
  } finally {
    btn.textContent = 'Reload'; btn.disabled = false;
  }
}

function renderReasoning() {
  if (!reasoningEntries || !reasoningEntries.length) {
    $('reasoning').innerHTML = '<div class="empty">No reasoning entries found.</div>';
    return;
  }
  $('reasoning').innerHTML = reasoningEntries.map((entry, i) => {
    const r = entry.reasoning || {};
    const summary = r.summary || {};
    const topTrades = r.top_trades || [];
    const tickTime = fmtTime(entry.tick_id);
    const submitted = summary.trades_submitted ?? topTrades.length;
    const evaluated = summary.markets_evaluated ?? 0;
    const equity = usd(summary.equity_usd);
    const exposure = usd(summary.planned_exposure_usd);

    return `<div class="rcard">
      <div class="rcard-hd" onclick="toggleReason(${i})">
        <div>
          <span class="tick-info">${esc(tickTime)}</span>
          <span class="tick-sub">${submitted} trades · ${evaluated} markets evaluated</span>
        </div>
        <span class="chev" id="chev-${i}">▼</span>
      </div>
      <div class="rcard-body hide" id="rbody-${i}">
        <div style="display:flex;gap:24px;margin-bottom:10px;flex-wrap:wrap">
          ${r.llm_weight != null ? `<div><span class="muted">llm_weight</span> <b style="color:var(--fg)">${r.llm_weight.toFixed(3)}</b></div>` : ''}
          ${r.backend != null ? `<div><span class="muted">backend</span> <b style="color:var(--fg)">${esc(r.backend)}</b></div>` : ''}
          ${summary.equity_usd != null ? `<div><span class="muted">equity</span> <b style="color:var(--fg)">${equity}</b></div>` : ''}
          ${summary.planned_exposure_usd != null ? `<div><span class="muted">new exposure</span> <b style="color:var(--fg)">${exposure}</b></div>` : ''}
          ${summary.risk_fraction != null ? `<div><span class="muted">risk_f</span> <b style="color:var(--fg)">${(summary.risk_fraction * 100).toFixed(1)}%</b></div>` : ''}
        </div>
        ${topTrades.length
        ? `<div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;margin-bottom:6px">Top Trades</div>
             ${topTrades.map(t => {
          const side = t.side || (t.action ? t.action.split(' ').pop() : '');
          const sideCls = side === 'YES' ? 'green' : side === 'NO' ? 'red' : 'muted';
          const edge = num(t.edge);
          return `<div class="rtrade">
                <div class="q">
                  <div>${esc(t.question || t.market_id)}</div>
                  <div class="mid">${esc(t.market_id)}</div>
                </div>
                <span class="${sideCls}">${esc(side)}</span>
                <span class="mono">${fmt(t.shares)} @ ${fmt(t.price, 3)}</span>
                <span class="mono green">${edge != null ? `+${(edge * 100).toFixed(1)}%` : '—'}</span>
              </div>`;
        }).join('')}`
        : '<div class="empty" style="padding:8px 0">No trades this tick.</div>'}
      </div>
    </div>`;
  }).join('');
}

function toggleReason(i) {
  const body = $('rbody-' + i), chev = $('chev-' + i);
  const nowHidden = body.classList.toggle('hide');
  chev.textContent = nowHidden ? '▼' : '▲';
}

// ── Poll loop ─────────────────────────────────────────────────────────────────

(async () => {
  try { await load(); } catch (e) {
    $('err').textContent = 'Error: ' + e.message;
    $('err').classList.remove('hide');
  }
  render();
  setInterval(async () => {
    try { await load(); render(); } catch (_) {}
  }, POLL_MS);
})();
</script>
</body>
</html>
"""
