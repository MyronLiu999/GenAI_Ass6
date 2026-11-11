"""EquityDesk: a multi-agent equity research desk on LangGraph, with a Streamlit front end.

Layers (see ARCHITECTURE.md):
  model + harness   equitydesk/harness/   metered LLM client, budgets, retries, pricing, traces
  tools             equitydesk/tools/     market data, indicators, risk, fundamentals, evidence pack
  memory & state    equitydesk/memory/    SQLite store (analyses, notes, profile, runs) + LangGraph checkpoints
  control & policy  equitydesk/policy/    input guards, untrusted-text handling, grounding check
  graph             equitydesk/graph/     the LangGraph desk: analysts -> debate -> PM -> grounding -> report
  evaluation        equitydesk/eval/      offline eval on recorded fixtures; track record = online eval
"""

__version__ = "1.0.0"
