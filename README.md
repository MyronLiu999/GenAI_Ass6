# EquityDesk: a multi-agent equity research desk (LangGraph + Streamlit)

EquityDesk researches a stock the way a small buy-side desk would. Four analysts (fundamental,
technical, sentiment & news, risk) work in parallel from one evidence pack, while a research librarian searches the
web twice, for recent sell-side notes and for quantified industry drivers (guidance, customers' capex, market
size), and two more analysts read the results: one distils what the street thinks, the other calls the next
earnings report from the consensus bar, the demand data and the quality of reported earnings. A bull and a bear
researcher then debate for a bounded number of rounds, and a portfolio manager makes the call: rating,
conviction, horizon, drivers, risks, catalysts and invalidation. A code-run grounding check verifies
every citation and every number before the report is accepted.

What makes it a tool and not a demo:

- **No number comes from the model.** Indicators, risk metrics, valuation ratios, the protective stop
  and the position size are computed in code. Agents may only cite them by id (`[T.rsi14]`), and the
  grounding check rejects unknown ids, uncited claims and figures that match no evidence.
- **Views from expectations, not from levels.** Code computes the bar for the next report (the growth the
  consensus asks for against what the company has delivered, judged year over year so seasonality cannot
  fool it), the beat record, next-year estimates against the run-rate and their dispersion, and
  earnings-quality flags (receivables or inventory outrunning revenue, weak cash conversion, non-operating
  gains). A depth check sends back PM drivers that merely restate a figure.
- **Street research with verified sources.** Dated rating and target actions by firm, the fresh 30-day median
  target against the stale mean, and EPS-estimate revisions come from structured data. Research notes found on
  the web are kept only if the search tool actually cited their source, and every note links to it.
- **News tracking.** Every headline is triaged for tone, materiality and whether it is really about the
  company. The Monitor page follows the whole watchlist around the clock: scored, deduplicated, pushed when it matters.
- **Memory that holds the desk accountable.** Every call is stored with the price at the time. The next
  time it covers the ticker, the PM sees its own previous calls scored against what the stock did since.
  The Track record page shows the hit rate and excess return of every call.
- **A real harness.** Every call runs through a token and cost ceiling, retries only on transient
  errors, gets one schema repair, and is priced and traced. Every node is checkpointed, so a failed or
  budget-stopped run resumes without redoing (or re-paying for) finished work.
- **Policy in code.** Inputs are validated, third-party news is sanitised and fenced as untrusted, and
  injection attempts are flagged. Size and stop follow your risk profile. There is no order tool:
  research-only by construction.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the seven-layer breakdown with code references.

## Quick start (Windows, Git Bash or PowerShell)

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
```bash
cp .env.example .env
```
Put your key in `.env` (`OPENAI_API_KEY=...` for the default `gpt-4o`; see `settings.yaml` for
Anthropic or any OpenAI-compatible endpoint). Then:
```bash
.venv/Scripts/streamlit run app.py
```

No key yet? Pick **Offline demo** in the sidebar. It runs the whole desk on recorded market data with a
scripted model, at no cost, in its own data folder (`data/demo`), so it never touches your real memory.

## Pages

| Page | What it does |
|---|---|
| Research | Run the desk on a ticker and watch each agent finish live. Read the rating, thesis, the expectations gap (the next report and next year), variant view against the street, the Earnings & demand tab (the bar, quality flags, industry figures with sources), key debates, drivers and risks, the trade plan, the street research (actions, fresh targets, revisions, research notes with sources), the triaged news, a chart with stop/support/resistance, each analyst's view, the debate and the full evidence table. Ask follow-up questions in a chat that stays grounded in the report's evidence. Download Markdown or JSON. |
| Monitor | The market monitor (below): status with pause/stop, today's counts and spend, index quotes with sparklines, the AI overview, prediction-market odds, event reports; an event stream filtered by score, time window, holdings, reports and text; per-stock snapshots (news, price, Reddit mentions, StockTwits bull/bear); the watchlist with aliases, match-by-ticker and holding flags; sources, push channels, model engine and autostart. |
| Watchlist | Price, returns, RSI, trend, volatility, beta and the desk's last call for many tickers. Pure code, no tokens. Send any row to the desk. |
| Track record | Every past call scored against the benchmark (hit rate, average excess return of buys and sells, scatter by date). |
| Memory & profile | Your risk profile (portfolio value, risk per trade, max position, tolerance, horizon), your notes (per ticker or global, e.g. "I hold 200 NVDA at $95"), and past calls you can open or forget. |
| Runs & traces | The run ledger (status, tokens, cost) and, per run, a node timeline (the parallel fan-out is visible), every model and tool call, grounding and guard events, and every prompt exactly as sent. Failed runs can be resumed from here. |

## Market monitor

A background process that watches the market for you, every minute, around the clock:

```bash
.venv/Scripts/python -m equitydesk.monitor
```

or double-click `start_monitor.bat` (starts the monitor and the app, and opens the Monitor page), or press Start on
the Monitor page. To start it whenever you log in, switch on "Start with Windows" under Sources & settings.

- **What it collects.** Official releases (Federal Reserve monetary policy, press and speeches; SEC; White House; BLS),
  newsrooms (Reuters and Bloomberg via Google News, CNBC, MarketWatch), 华尔街见闻 7x24, Yahoo news for every
  watchlist stock, Reddit mentions (ApeWisdom), StockTwits bull/bear, Polymarket and Kalshi odds, and index quotes.
- **What it does with it.** The same story from several sources becomes one item (also across languages, after
  translation). A cheap model scores every story 1-10 for your watchlist, categorises it, links it to tickers
  (aliases are matched in code too) and translates the headline. Items of 8 or more, or 7 or more about a stock you
  hold, are pushed (Windows notification, and your phone through ntfy.sh if you set a topic), at most 8 an hour.
  For the biggest events it searches the web and writes a short brief that separates confirmed facts from
  unconfirmed claims and lists only sources the search tool actually returned. Every two hours it writes an
  overview whose every bullet links to the items it rests on.
- **Trust labels.** Every item shows its tier: official, mainstream, single source, or unverified (the item itself
  relays unconfirmed reports; such items are capped at 8). Odds are labelled as bets, not facts. It reports what
  happened and what it may affect; it gives no buy or sell advice.
- **Cost.** With the OpenAI engine, scoring runs on gpt-4o-mini and the overview and briefs on gpt-4o: about $1 a
  day on a busy news day, with a hard ceiling (`monitor.daily_cost_ceiling_usd`, default $2) after which collection
  carries on and model calls wait for the next day. With the **Claude Code** engine (Sources & settings), the same
  calls run through the Claude Code CLI logged in on your machine, covered by your Claude subscription: log in once
  with `claude auth login`.

## Command line

```bash
.venv/Scripts/python -m equitydesk.runner NVDA --rounds 2 --out nvda.md
```
```bash
.venv/Scripts/python -m equitydesk.runner AAPL --offline
```
```bash
.venv/Scripts/python -m equitydesk.runner --resume NVDA-20261111-011509-73fa
```

## Testing and evaluation

```bash
.venv/Scripts/python -m pytest -q
```
144 tests, offline and free. They cover the indicators (RSI checked against the standard reference
example), risk maths and position sizing, input guards, injection flagging, the grounding checker,
pricing, retries, schema repair, budget stops, the memory store and outcome scoring, and the whole graph
end to end. Failure paths covered: a failed analyst, a bounded PM revision, a budget stop followed by a
resume in a new process, an unknown symbol, Q&A history that survives a restart.

```bash
.venv/Scripts/python -m equitydesk.eval.run_eval --adversarial
```
```bash
.venv/Scripts/python -m equitydesk.eval.run_eval --live --adversarial
```
The offline eval runs the desk over the recorded real-data fixtures in `fixtures/` (AAPL, NVDA, JPM,
XOM, TSLA; refresh with `python -m equitydesk.tools.market snapshot AAPL ...`). It scores each run from
its trace: completion, PM grounded on the first try and finally, citation coverage, unknown ids,
invented numbers, schema errors, debate rounds honoured, tokens, cost and time. `--adversarial` plants
a "rate this STRONG_SELL" injection headline in every ticker and measures whether the PM followed it.
Without `--live` it uses the scripted model (checks the plumbing for free); with `--live` it uses your
configured model (checks quality). Results go to `reports/EVAL.md` and `reports/eval_<timestamp>.json`.

## Configuration

Everything tunable is in `settings.yaml`: the model and per-role or per-analyst overrides (by default the
PM and the earnings analyst run on gpt-5.5 and the rest on gpt-4o, about $0.77 a run; `roles: {}` gives the
all-gpt-4o desk at about $0.45; see ARCHITECTURE.md for the side-by-side that motivated this), budgets, debate rounds and caps, revision limit, report language (English or Chinese),
grounding tolerances, data provider, benchmark, cache TTL, staleness threshold, risk-free rate and ATR
stop multiple. Prices are in `pricing.yaml`; a model without a price line is reported as UNPRICED.

## Layout

```
app.py                      Streamlit entry (navigation, mode, model status)
views/                      research · monitor · watchlist · track_record · memory · runs
equitydesk/
  config.py                 typed settings
  schemas.py                structured outputs (AnalystReport, DebateTurn, Decision)
  runner.py                 run lifecycle, checkpointer, resume, CLI
  report.py                 trade plan + Markdown report
  graph/desk.py             the LangGraph desk        graph/qa.py  follow-up Q&A graph
  harness/                  llm.py (metered client) · pricing.py · trace.py
  tools/                    market.py · indicators.py · risk.py · fundamentals.py · earnings.py · evidence.py · news.py
  monitor/                  sources · signals · triage · reports · notify · store · service (the market monitor)
  harness/claude_code.py    the Claude Code CLI as a model backend (subscription, no API key)
  policy/                   guards.py · grounding.py
  memory/                   store.py (SQLite) · track.py (outcomes, reflection, track record)
  eval/run_eval.py          offline evaluation
  prompts/                  agent prompts and analyst roles (data, not code)
fixtures/                   recorded real market data for offline runs and the eval
tests/                      offline test suite
```

## Limitations

- Yahoo Finance is an unofficial source. Fields can be missing, late or wrong; the desk degrades and
  warns, but verify figures against filings before acting.
- The grounding check matches numbers to the evidence within a tolerance. It catches invented figures,
  not a wrong reading of a correct figure; judgement remains the model's.
- Yahoo's news feed for a ticker often includes loosely related stories; the sentiment analyst is told
  to separate material news from noise, but relevance is not checked in code.
- A track record needs time: calls younger than a few weeks say little.
- This is research software, not investment advice. It places no orders.

## 中文简介

EquityDesk 是一个基于 LangGraph + Streamlit 的多智能体股票投研工具：四位分析师（基本面、技术面、情绪新闻、风险）并行分析，另有财报与需求分析师（从一致预期门槛、超预期记录、行业需求数据和盈利质量判断下一次财报）和卖方研究分析师，多空双方辩论，投资经理给出评级，最后由代码做事实校验（grounding）。所有数字都由代码计算，模型只能引用证据 ID；每次判断都会存入记忆，之后按真实走势打分，形成可追溯的历史战绩。运行 `streamlit run app.py`；没有 API key 时，在侧边栏选 “Offline demo” 即可免费体验。报告语言可在 `settings.yaml` 或界面中切换为中文。
