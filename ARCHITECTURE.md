# ARCHITECTURE.md · EquityDesk on the seven layers

For each layer: what does the job, and where it is enforced in code. A layer held only by a sentence
in a prompt (any file under `equitydesk/prompts/`) would count as **EMPTY**. References are
`path:line`.

| Layer | What does this job | Where it is enforced | Status |
|---|---|---|---|
| Model | One metered client for every call; OpenAI, any OpenAI-compatible endpoint, or Anthropic through LangChain; per-role model overrides (analyst / debate / pm / qa); an offline scripted model for tests and demo mode | `equitydesk/harness/llm.py:276`, `equitydesk/harness/llm.py:106`, `equitydesk/harness/llm.py:154`, `equitydesk/config.py:74` | present |
| Harness | The LangGraph StateGraph (fan-out of four analysts, a join, a bounded bull/bear loop, a bounded PM revision loop); a token and cost ceiling checked before every call; retries with backoff on transient errors only; auth/not-found errors stop the run; one schema-repair attempt; the recursion limit as a backstop; Q&A history trimmed to the last 12 messages | `equitydesk/graph/desk.py:99`, `equitydesk/graph/desk.py:308`, `equitydesk/graph/desk.py:216`, `equitydesk/graph/desk.py:260`, `equitydesk/harness/llm.py:298`, `equitydesk/harness/llm.py:322`, `equitydesk/harness/llm.py:325`, `equitydesk/harness/llm.py:349`, `equitydesk/runner.py:30`, `equitydesk/graph/qa.py:37` | present |
| Tools | Market data (Yahoo Finance, disk-cached, fetched in parallel, each component degrading on its own) or recorded fixtures; indicators, risk metrics, fundamentals and the ATR trade plan as pure functions; the evidence pack that gives every number a citable id. No order-placement tool exists: research-only by construction | `equitydesk/tools/market.py:156`, `equitydesk/tools/market.py:215`, `equitydesk/tools/market.py:126`, `equitydesk/tools/market.py:226`, `equitydesk/tools/risk.py:63`, `equitydesk/tools/evidence.py:113`, `equitydesk/report.py:9` | present |
| Memory & State | Short-term: the graph state checkpointed after every node (SqliteSaver), so a failed or budget-stopped run resumes without redoing finished nodes; each report's Q&A thread checkpointed under its own thread id. Long-term: a SQLite store of past calls with the price at the time, the user's notes (scoped, soft-deleted), the risk profile and the run ledger. Reflection: past calls on a ticker are scored against what happened and shown to the PM | `equitydesk/runner.py:48`, `equitydesk/runner.py:80`, `equitydesk/graph/qa.py:18`, `equitydesk/memory/store.py:51`, `equitydesk/memory/store.py:100`, `equitydesk/memory/store.py:143`, `equitydesk/graph/desk.py:151`, `equitydesk/memory/track.py:16`, `equitydesk/memory/track.py:44`, `equitydesk/graph/desk.py:285` | present |
| Control & Policy | Tickers validated before anything runs; debate rounds clamped; third-party text sanitised, fenced in `<untrusted>` and flagged when instruction-like; every agent output grounding-checked (citations exist, claims cited, numbers match the evidence), and a failed PM decision sent back with the problems listed; consistency check of the PM against the analysts; position size and stop computed in code from the user's risk budget; the disclaimer added by code | `equitydesk/policy/guards.py:31`, `equitydesk/config.py:109`, `equitydesk/policy/guards.py:38`, `equitydesk/policy/guards.py:13`, `equitydesk/tools/evidence.py:275`, `equitydesk/policy/grounding.py:130`, `equitydesk/policy/grounding.py:163`, `equitydesk/graph/desk.py:247`, `equitydesk/policy/grounding.py:177`, `equitydesk/report.py:9`, `equitydesk/policy/guards.py:22` | present |
| Observability | One JSONL trace per run: node spans, every model call (tokens, cost, latency, attempt, schema errors), every tool call (cached or not), grounding results, guard events, memory reads and writes, budget stops; every prompt logged, numbered, exactly as sent; the run ledger; the Runs & traces page (timeline, per-node cost, prompt viewer) | `equitydesk/harness/trace.py:21`, `equitydesk/harness/trace.py:60`, `equitydesk/harness/trace.py:51`, `equitydesk/harness/llm.py:320`, `equitydesk/graph/desk.py:104`, `equitydesk/memory/store.py:71`, `views/runs.py` | present |
| Evaluation | 69 offline tests (no network, no tokens) including failure paths; an offline eval over recorded real-data fixtures that scores runs from their traces (grounding, citation coverage, invented numbers, schema errors, cost), with an adversarial mode that plants an injection headline; the track record scores every live call against the benchmark: evaluation in production | `tests/`, `equitydesk/eval/run_eval.py:54`, `equitydesk/eval/run_eval.py:38`, `equitydesk/memory/track.py:69`, `equitydesk/memory/track.py:80`, `views/track_record.py` | present |

**Empty layers: 0 of 7.**

## The graph

```
START -> collect_data --(no usable prices)---------------------------------------------> finalize -> END
              |
              v
        recall_memory -> fundamental ┐
                      -> technical   ├── join ──> bull -> bear --(round < N)--> bull
                      -> sentiment   │                     |
                      -> risk        ┘                     v (round == N)
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
| Empty memory note; one-sided invalidation | Schema: `memory_note` required, `invalidation` at least two items |
| Citations dropped from the text or written as `[A, B]` | Normalised in code after every agent call; numbers in invalidation and catalysts are checked too |

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
