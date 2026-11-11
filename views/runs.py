"""Runs & traces: the ledger of every run, and a step-by-step record of what each one did."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from equitydesk.harness.trace import read_prompts, read_trace
from equitydesk.ui.common import NODE_LABEL, desk, open_report

d = desk()
st.title("Runs & traces")

runs = d.store.list_runs()
if not runs:
    st.info("No runs yet.")
    st.stop()

ledger = pd.DataFrame(runs)
c = st.columns(4)
c[0].metric("Runs", len(ledger))
c[1].metric("Completed", int((ledger["status"] == "completed").sum()))
c[2].metric("Tokens", f"{int(ledger['tokens'].sum()):,}")
c[3].metric("Spend", f"${ledger['cost_usd'].sum():.3f}")

ev = st.dataframe(ledger[["started_at", "run_id", "ticker", "status", "model", "debate_rounds", "tokens", "cost_usd",
                          "error"]], hide_index=True, on_select="rerun", selection_mode="single-row", key="ledger",
                  column_config={"cost_usd": st.column_config.NumberColumn("cost", format="$%.4f"),
                                 "tokens": st.column_config.NumberColumn(format="%d")})
picked = ev.selection.rows[0] if ev and ev.selection and ev.selection.rows else 0
ids = [r["run_id"] for r in runs]
choice = st.selectbox("Inspect run", ids, index=picked,
                      format_func=lambda i: f"{i} · {next(r['status'] for r in runs if r['run_id'] == i)}",
                      help="Defaults to the latest run, or the row selected above.")
run = next(r for r in runs if r["run_id"] == choice)
rid = run["run_id"]
trace = read_trace(rid, d.s.runs_dir)
st.subheader(rid)
b = st.columns(3)
if run["status"] in ("completed",) and b[0].button("Open report", icon=":material/open_in_new:", width="stretch"):
    open_report(rid)
if run["status"] not in ("completed", "no_data", "running"):
    if b[1].button("Resume from checkpoint", icon=":material/replay:", type="primary", width="stretch"):
        with st.status(f"Resuming {rid}…", expanded=True) as status:
            for e in d.stream(rid, resume=True):
                if e["type"] == "node":
                    status.write(f":material/check_circle: {NODE_LABEL.get(e['node'], e['node'])}")
                elif e["type"] == "error":
                    status.update(label=f"Stopped again: {e['error']}", state="error")
                else:
                    status.update(label="Completed", state="complete")
                    st.session_state["report"] = e["report"]
        st.rerun()

# node timeline: parallel fan-out shows as overlapping bars (follow-up Q&A happens later; it is listed below)
spans, open_ = [], {}
for e in trace:
    if e.get("node") == "qa":
        continue
    if e["type"] == "node_start":
        open_[e["node"]] = e["t"]
    elif e["type"] == "node_end" and e["node"] in open_:
        t0 = open_.pop(e["node"])
        spans.append({"node": NODE_LABEL.get(e["node"], e["node"]), "start": datetime.fromtimestamp(t0),
                      "end": datetime.fromtimestamp(max(e["t"], t0 + 0.01)), "ms": e.get("ms"),
                      "ok": e.get("ok")})
if spans:
    sp = pd.DataFrame(spans)
    fig = px.timeline(sp, x_start="start", x_end="end", y="node", color="ok",
                      color_discrete_map={True: "#22c55e", False: "#ef4444"}, hover_data=["ms"])
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(height=60 + 28 * sp["node"].nunique(), margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, key="timeline")

llm = pd.DataFrame([e for e in trace if e["type"] == "llm_call"])
tools = pd.DataFrame([e for e in trace if e["type"] == "tool_call"])
guards = pd.DataFrame([e for e in trace if e["type"] in ("grounding", "guard", "budget_exceeded", "memory_read",
                                                          "memory_write", "run_start", "run_resume", "run_end")])
t1, t2, t3, t4, t5 = st.tabs([f"Model calls ({len(llm)})", f"Tool calls ({len(tools)})",
                              f"Checks & events ({len(guards)})", "Prompts", "Raw trace"])
with t1:
    if not llm.empty:
        cols = [c for c in ["ts", "node", "model", "request", "attempt", "ok", "repair", "ms", "input_tokens",
                            "output_tokens", "cost_usd", "error", "schema_error"] if c in llm]
        st.dataframe(llm[cols], hide_index=True,
                     column_config={"cost_usd": st.column_config.NumberColumn(format="$%.5f")})
        by = llm.groupby("node").agg(calls=("node", "size"), input_tokens=("input_tokens", "sum"),
                                     output_tokens=("output_tokens", "sum"), cost_usd=("cost_usd", "sum"),
                                     ms=("ms", "sum")).reset_index()
        st.caption("By node")
        st.dataframe(by, hide_index=True, column_config={"cost_usd": st.column_config.NumberColumn(format="$%.5f")})
with t2:
    if not tools.empty:
        st.dataframe(tools[[c for c in ["ts", "tool", "ticker", "ok", "cached", "ms", "error"] if c in tools]],
                     hide_index=True)
with t3:
    if not guards.empty:
        st.dataframe(guards.drop(columns=[c for c in ("run_id", "t") if c in guards]), hide_index=True)
with t4:
    prompts = read_prompts(rid, d.s.runs_dir)
    st.caption(f"{len(prompts)} prompts, exactly as sent.")
    for p in prompts:
        with st.expander(f"#{p['request']} · {p['node']} · {p['model']}"):
            for msg in p["messages"]:
                st.markdown(f"**{msg['role']}**")
                st.code(msg["content"] if isinstance(msg["content"], str) else str(msg["content"]), language=None,
                        wrap_lines=True)
with t5:
    st.json(trace, expanded=False)
