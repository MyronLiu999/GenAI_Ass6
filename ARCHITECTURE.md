# ARCHITECTURE.md · EquityDesk on the seven layers

For each layer: what does the job, and where it is enforced in code. A layer held only by a sentence
in a prompt (any file under `equitydesk/prompts/`) would count as **EMPTY**. References are
`path:line`.

| Layer | What does this job | Where it is enforced | Status |
|---|---|---|---|
| Model | One metered client for every call; OpenAI, any OpenAI-compatible endpoint, or Anthropic through LangChain; per-role model overrides (analyst / debate / pm / qa / research) and per-node ones (e.g. only the earnings analyst); reasoning models (gpt-5.x) through the Responses API with a reasoning effort; OpenAI's hosted web-search tool, metered like any call with its per-call fee; an offline scripted model for tests and demo mode | `equitydesk/harness/llm.py:392`, `equitydesk/harness/llm.py:114`, `equitydesk/harness/llm.py:201`, `equitydesk/config.py:85` | present |
| Harness | The LangGraph StateGraph (fan-out of four analysts plus a web-research node, running two searches in parallel, that feeds a fifth and a sixth analyst, a join, a bounded bull/bear loop, a bounded PM revision loop); a token and cost ceiling checked before every call; retries with backoff on transient errors only; auth/not-found errors stop the run; one schema-repair attempt; the recursion limit as a backstop; Q&A history trimmed to the last 12 messages | `equitydesk/graph/desk.py:115`, `equitydesk/graph/desk.py:443`, `equitydesk/graph/desk.py:319`, `equitydesk/graph/desk.py:390`, `equitydesk/harness/llm.py:414`, `equitydesk/harness/llm.py:440`, `equitydesk/harness/llm.py:443`, `equitydesk/harness/llm.py:470`, `equitydesk/runner.py:30`, `equitydesk/graph/qa.py:37` | present |
| Tools | Market data (Yahoo Finance, disk-cached, fetched in parallel, each component degrading on its own) or recorded fixtures; sell-side research data (dated rating and target actions by firm, target range, EPS estimate trend and revisions, recommendation mix) turned into A.* facts; the earnings set-up (beat record, the consensus bar against the run-rate, quarter and next-year estimates with their dispersion) as E.* facts and reported quarterly results with earnings-quality checks as Q.* facts (`equitydesk/tools/earnings.py`); web search for research notes, kept as W.* facts, and for quantified industry and demand drivers (guidance, customers' capex, market size, supply), kept as D.* facts; the news tracker; indicators, risk metrics, fundamentals and the ATR trade plan as pure functions; the evidence pack that gives every number a citable id. No order-placement tool exists: research-only by construction | `equitydesk/tools/market.py:282`, `equitydesk/tools/market.py:360`, `equitydesk/tools/market.py:252`, `equitydesk/tools/market.py:372`, `equitydesk/tools/risk.py:63`, `equitydesk/tools/evidence.py:120`, `equitydesk/report.py:9` | present |
| Memory & State | Short-term: the graph state checkpointed after every node (SqliteSaver), so a failed or budget-stopped run resumes without redoing finished nodes; each report's Q&A thread checkpointed under its own thread id. Long-term: a SQLite store of past calls with the price at the time, the user's notes (scoped, soft-deleted), the risk profile and the run ledger. Reflection: past calls on a ticker are scored against what happened and shown to the PM | `equitydesk/runner.py:48`, `equitydesk/runner.py:82`, `equitydesk/graph/qa.py:18`, `equitydesk/memory/store.py:56`, `equitydesk/memory/store.py:105`, `equitydesk/memory/store.py:148`, `equitydesk/graph/desk.py:167`, `equitydesk/memory/track.py:16`, `equitydesk/memory/track.py:44`, `equitydesk/graph/desk.py:417` | present |
| Control & Policy | Tickers validated before anything runs; debate rounds clamped; third-party text (news, web research) sanitised, fenced in `<untrusted>` and flagged when instruction-like; web research keeps only notes whose sources the search tool actually cited (`equitydesk/policy/provenance.py`); third-party text escaped before it is rendered as HTML (`equitydesk/ui/components.py`); every agent output grounding-checked (citations exist, claims cited, numbers match the evidence), and a failed PM decision sent back with the problems listed; consistency check of the PM against the analysts; position size and stop computed in code from the user's risk budget; the disclaimer added by code | `equitydesk/policy/guards.py:31`, `equitydesk/config.py:123`, `equitydesk/policy/guards.py:38`, `equitydesk/policy/guards.py:13`, `equitydesk/tools/evidence.py:439`, `equitydesk/policy/grounding.py:143`, `equitydesk/policy/grounding.py:182`, `equitydesk/graph/desk.py:350`, `equitydesk/policy/grounding.py:243`, `equitydesk/report.py:9`, `equitydesk/policy/guards.py:22` | present |
| Observability | One JSONL trace per run: node spans, every model call (tokens, cost, latency, attempt, schema errors), every tool call (cached or not), grounding results, guard events, memory reads and writes, budget stops; every prompt logged, numbered, exactly as sent; the run ledger; the Runs & traces page (timeline, per-node cost, prompt viewer) | `equitydesk/harness/trace.py:21`, `equitydesk/harness/trace.py:60`, `equitydesk/harness/trace.py:51`, `equitydesk/harness/llm.py:438`, `equitydesk/graph/desk.py:120`, `equitydesk/memory/store.py:76`, `views/runs.py` | present |
| Evaluation | 144 offline tests (no network, no tokens) including failure paths; an offline eval over recorded real-data fixtures that scores runs from their traces (grounding, citation coverage, invented numbers, schema errors, cost), with an adversarial mode that plants an injection headline; the track record scores every live call against the benchmark: evaluation in production | `tests/`, `equitydesk/eval/run_eval.py:54`, `equitydesk/eval/run_eval.py:38`, `equitydesk/memory/track.py:69`, `equitydesk/memory/track.py:80`, `views/track_record.py` | present |

**Empty layers: 0 of 7.**

## The graph

```
START -> collect_data --(no usable prices)---------------------------------------------> finalize -> END
              |
              v
        recall_memory -> fundamental ──────────────┐
                      -> technical   ──────────────┤
                      -> sentiment   ──────────────├── join ──> bull -> bear --(round < N)--> bull
                      -> risk        ──────────────┤                     |
                      -> web_research -> street ───┤                     v (round == N)
                                      -> earnings ─┘
                                                                        pm -> grounding --(failed and revisions left)--> pm
                                                                    |
                                                                    v
                                                                 finalize -> END
```

* State is plain JSON (`equitydesk/graph/state.py`); nodes are closures over a `Runtime` holding the
  provider, the metered LLM, the tracer and the store, so nothing unserialisable is ever checkpointed.
* Parallel analysts write into one dict through a merge reducer; warnings and debate turns accumulate
  through `operator.add`.
* Both loops are bounded in code: the debate by `graph.max_debate_rounds`, the PM rewrite by
  `graph.max_revisions`. The LangGraph recursion limit (80) is only a backstop.

## Web research and provenance

`web_research` makes two calls. The first is OpenAI's hosted web search, run as plain text so that it returns
`url_citation` annotations; each cited span is replaced by a marker `[S<n>]` bound to the real URL. The
second extracts `ResearchNotes`, and every note must name its markers. `verify_notes` drops any note whose
markers do not resolve to a cited URL, and flags notes that come only from ratings-table aggregators. The
kept notes become `W.*` facts (untrusted, sanitised, injection-flagged), and the Street Research analyst
reads them together with the structured `A.*` record. Asking for structured output *during* the search
was tested live: it returned no citations and invented its sources, which is why the search is split in two.

## Expectations, not levels: the earnings set-up and the depth check

A first live version restated figures ("revenue grew 105.9%, showing strong momentum"). The market already
knows the level of every reported figure; a view is about where reality will differ from what is priced in.
Two changes made that possible, and one check enforces it.

* **Data the model could not argue from before.** `equitydesk/tools/earnings.py` computes the bar for the
  next report: the growth the consensus asks for, set against what the company has delivered. Year-over-year
  growth decides whether the bar is high or low, because it is free of seasonality; the sequential view is
  shown beside it, and for uneven or seasonal businesses (a holiday quarter, an oil-price swing) the
  sequential run-rate is not used. It also computes the beat record, two run-rate scenarios (labelled as
  arithmetic, not forecasts), next-year consensus against the annualised last quarter, estimate dispersion,
  and earnings-quality checks: receivables and inventory growing faster than revenue (year over year, so a
  launch-quarter build is not a warning), days sales outstanding, cash conversion, non-operating gains in net
  income. Banks skip the working-capital checks.
* **Industry figures with sources.** A second hosted search asks for quantified drivers (the company's
  guidance, its largest customers' capex plans, market size, supply, pricing, competition). Each figure must
  carry its source marker and survive the same provenance check as the research notes; kept figures become
  untrusted `D.*` facts. Numbers quoted from them are grounded like any other.
* **The depth check** (`shallow_drivers`, `equitydesk/policy/grounding.py`). A PM driver that cites fewer than
  two facts, facts of a single category, or reads as a restatement ("showing strong ...", "显示出强劲") is
  shallow. If most drivers are shallow, or `expectations_gap` is empty while E.* facts exist, the decision goes
  back to the PM with the drivers quoted. Depth never fails the grounding check itself: it costs a revision,
  and if the revision is still shallow the report says so.

The Earnings & Demand analyst reads E, Q and D and must call the next report (beat likely, in line, miss risk).

## Which model, and why

Same evidence, same analysts, NVDA in Chinese, 2026-11-11. With a gpt-4o PM, two of the drivers still rested on
volatility and on the stale mean target, and the final decision failed the stale-estimate check. With a
gpt-5.5 PM (reasoning effort medium), the thesis argued the bar for the 2026-11-17 report (guidance of $108B
±2% against a $109.0B consensus, a run-rate 5.7% above it, eight straight beats) and set it against the
receivables and cash-conversion flags and the governance risk; it passed every check after one revision.
The gpt-4o earnings analyst also misread $108B guidance as above a $109.0B consensus, a comparison the
grounding check cannot catch. Default since then: PM and earnings analyst on gpt-5.5, the other analysts,
the debate and the research extraction on gpt-4o; about $0.77 per run instead of $0.45. Better data and
checks made the views possible; the stronger model is what makes the reasoning over them reliable.

## Why numbers never come from the model

Every figure an agent may use is computed by a pure function and enters the prompt as a cited fact
(`[T.rsi14] RSI(14): 55.62`). The grounding check (`equitydesk/policy/grounding.py`) extracts every
amount, percentage and multiple from what an agent wrote and matches it against the evidence values
within a tolerance; bare integers that name periods or parameters (`200-day`, `RSI(14)`, `12m`) are
skipped. Position size and stop are computed from the user's profile (`equitydesk/report.py:9`); the PM
can only refer to them.

## Judgement rules enforced in code (learned from live gpt-4o runs)

| Observed in a live run | Rule now in code |
|---|---|
| "Golden cross" label while the price sat 22% below both averages | `T.trend` combines the cross with where the price sits (`trend_regime` in `equitydesk/tools/evidence.py`) |
| Oversold RSI read as "further downside" | `T.rsi_zone` states what the zone means |
| A −25% month with a +93% upside to target: estimates stale, yet used as a driver | `S.consensus_check` fact, plus a grounding rule: while it is present, a PM driver may not rest on forward P/E, PEG, targets or the consensus rating (`ESTIMATE_IDS`, `equitydesk/policy/grounding.py`) |
| Earnings in 6 days, no mention of gap risk | `S.event_risk` fact and report warning when earnings are within 14 days |
| A 15-minute-old call scored as a "hit" in memory | Calls younger than `grounding.min_days_to_score` are "too early" |
| Barred from the drivers, the PM moved the stale-estimate argument into a key debate | Any sentence in the thesis, variant view or debates that cites those estimates must carry a caveat (`CAVEAT`, `equitydesk/policy/grounding.py`) |
| Thesis opened with "presents a mixed investment case" | Boilerplate detector (`generic_phrases`): reported as a warning, and added to the revision feedback |
| Six of eight research notes came from one ratings-table page | The librarian prompt asks for the articles that explain each major action; aggregator-only notes are labelled |
| Drivers restated single figures ("毛利率74.7%，显示出其卓越的盈利能力") | E/Q/D facts to argue from, and the depth check (`shallow_drivers`); `expectations_gap` required when estimates exist |
| A sequential "bar" of +36.6% for Apple's holiday quarter, against a 5.2% average | The bar is judged year over year; the sequential run-rate is dropped when sequential growth is uneven |
| A 40% sequential receivables build at Apple's launch quarter flagged as a quality issue | Receivables and inventory are compared with revenue year over year |
| The stale-estimate caveat in a Chinese report ("可能滞后") was not recognised | `CAVEAT` and the sentence splitter understand Chinese |
| Empty memory note; one-sided invalidation | Schema: `memory_note` required, `invalidation` at least two items |
| Citations dropped from the text or written as `[A, B]` | Normalised in code after every agent call; numbers in invalidation and catalysts are checked too |

## The market monitor (`equitydesk/monitor/`)

A second agent system beside the desk: a long-running service rather than a graph run.

| Layer | In the monitor |
|---|---|
| Model | `monitor` role (gpt-4o-mini) scores and translates; `monitor_overview` (gpt-4o) writes the overview; `monitor_report` (the research role) writes web-searched briefs. The `claude_code` engine routes all three through the Claude Code CLI (`equitydesk/harness/claude_code.py`): `--json-schema` for structured output, only WebSearch/WebFetch for briefs |
| Harness | `MonitorService.tick` (collect in parallel, cluster, score, push, one brief, signals, overview); every source isolated; a bad tick is logged and the loop carries on; a single-instance lock; a daily cost ceiling that stops model calls but not collection; a cap on items scored per tick |
| Tools | RSS/Atom and JSON sources with a tier each (`sources.py`); quotes, ApeWisdom, StockTwits, Polymarket, Kalshi (`signals.py`) |
| Memory & State | Its own SQLite file in WAL mode (`store.py`): items with clusters and scores, reports, the watchlist with aliases and holding flags, and key/value state (control, heartbeat, signals, overview, source health, daily spend). The UI and the service share it across processes |
| Control & Policy | Run / pause / stop through the store; third-party text sanitised and injection-flagged before any model sees it; tickers matched in code from aliases; rumour items capped at 8 and labelled unverified; overview bullets without a valid item citation dropped; briefs keep only sources the search tool returned; push window of 2 hours, hourly cap, no backlog burst on the first tick; notification text passed to PowerShell through environment variables, never the command line |
| Observability | One trace file a day (`MON-YYYYMMDD`), source health on the Sources tab, the push log, the last tick's summary |
| Evaluation | `tests/test_monitor.py`: parsing, clustering (including across languages), matching, rumour cap, the tick end to end with fake sources, push caps and warm start, the cost ceiling, pause/stop, the Claude Code backend's provenance rule and its not-logged-in error |

Live findings that became code: Polymarket lists untraded placeholder outcomes at 50% (dropped unless traded);
the same story reworded by two outlets, or in English and Chinese, escaped title clustering (merged again after
translation, at a similarity threshold checked against real titles); the first tick pushed 14 backlog items
(warm start, 2-hour window, hourly cap); started from the app's Start button, the monitor died whenever the app server restarted, because a process-tree kill follows parent ids (it is now created through WMI, outside any app's tree, and its status checks that the process is really alive, not only that the heartbeat is fresh); gpt-4o-mini "translated" a CoreWeave headline into the Nvidia story behind it and scored it 9 (the prompt now requires the headline's own subject, and a company story that names no watchlist company is capped at 7).

## Failure behaviour

| Failure | What happens |
|---|---|
| One data component down (news, statements, ...) | That component is skipped, a warning goes into the report |
| No usable price history | No model is called; a `no_data` report explains why |
| One analyst's model call fails (non-transient) | That analyst is marked failed; the desk carries on and says so |
| Transient model error (429, 5xx, timeout) | Retried with exponential backoff, up to `model.max_retries` |
| Bad API key, no access, unknown model | The run stops at once with a configuration error (no silent degradation) |
| Invalid structured output | One repair call naming the validation error; then a model error |
| Token or cost ceiling reached | The run stops before the next call; resume continues from the checkpoint |
| PM cites unknown ids / invents numbers | Sent back with the problems listed, at most `graph.max_revisions` times; a still-failing decision is shown as FAILED |
