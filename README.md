# EquityDesk: a multi-agent equity research desk (LangGraph + Streamlit)

EquityDesk researches a stock the way a small buy-side desk would. Four analysts (fundamental,
technical, sentiment & news, risk) work in parallel from one evidence pack, while a research librarian searches the
web for recent sell-side notes and independent research, and a fifth analyst distils what the street
thinks. A bull and a bear
researcher then debate for a bounded number of rounds, and a portfolio manager makes the call: rating,
conviction, horizon, drivers, risks, catalysts and invalidation. A code-run grounding check verifies
every citation and every number before the report is accepted.

What makes it a tool and not a demo:

- **No number comes from the model.** Indicators, risk metrics, valuation ratios, the protective stop
  and the position size are computed in code. Agents may only cite them by id (`[T.rsi14]`), and the
  grounding check rejects unknown ids, uncited claims and figures that match no evidence.
- **Street research with verified sources.** Dated rating and target actions by firm, the fresh 30-day median
  target against the stale mean, and EPS-estimate revisions come from structured data. Research notes found on
  the web are kept only if the search tool actually cited their source, and every note links to it.
- **News tracking.** Every headline is triaged for tone, materiality and whether it is really about the
  company. The News page follows the whole watchlist, marks what is new, and lists analyst actions.
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
| Research | Run the desk on a ticker and watch each agent finish live. Read the rating, thesis, variant view against the street, key debates, drivers and risks, the trade plan, the street research (actions, fresh targets, revisions, research notes with sources), the triaged news, a chart with stop/support/resistance, each analyst's view, the debate and the full evidence table. Ask follow-up questions in a chat that stays grounded in the report's evidence. Download Markdown or JSON. |
| News | Headlines across the watchlist with NEW marks, a code filter for headlines that are not about the company, one-click AI triage (tone, materiality, why; about $0.01), and every sell-side action of the past weeks. |
| Watchlist | Price, returns, RSI, trend, volatility, beta and the desk's last call for many tickers. Pure code, no tokens. Send any row to the desk. |
| Track record | Every past call scored against the benchmark (hit rate, average excess return of buys and sells, scatter by date). |
| Memory & profile | Your risk profile (portfolio value, risk per trade, max position, tolerance, horizon), your notes (per ticker or global, e.g. "I hold 200 NVDA at $95"), and past calls you can open or forget. |
| Runs & traces | The run ledger (status, tokens, cost) and, per run, a node timeline (the parallel fan-out is visible), every model and tool call, grounding and guard events, and every prompt exactly as sent. Failed runs can be resumed from here. |

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
100 tests, offline and free. They cover the indicators (RSI checked against the standard reference
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

Everything tunable is in `settings.yaml`: the model and per-role overrides (for example, analysts on a
cheaper model), budgets, debate rounds and caps, revision limit, report language (English or Chinese),
grounding tolerances, data provider, benchmark, cache TTL, staleness threshold, risk-free rate and ATR
stop multiple. Prices are in `pricing.yaml`; a model without a price line is reported as UNPRICED.

## Layout

```
app.py                      Streamlit entry (navigation, mode, model status)
views/                      research · watchlist · track_record · memory · runs
equitydesk/
  config.py                 typed settings
  schemas.py                structured outputs (AnalystReport, DebateTurn, Decision)
  runner.py                 run lifecycle, checkpointer, resume, CLI
  report.py                 trade plan + Markdown report
  graph/desk.py             the LangGraph desk        graph/qa.py  follow-up Q&A graph
  harness/                  llm.py (metered client) · pricing.py · trace.py
  tools/                    market.py · indicators.py · risk.py · fundamentals.py · evidence.py
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

EquityDesk 是一个基于 LangGraph + Streamlit 的多智能体股票投研工具：四位分析师（基本面、技术面、情绪新闻、风险）并行分析，多空双方辩论，投资经理给出评级，最后由代码做事实校验（grounding）。所有数字都由代码计算，模型只能引用证据 ID；每次判断都会存入记忆，之后按真实走势打分，形成可追溯的历史战绩。运行 `streamlit run app.py`；没有 API key 时，在侧边栏选 “Offline demo” 即可免费体验。报告语言可在 `settings.yaml` 或界面中切换为中文。
