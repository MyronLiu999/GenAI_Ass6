"""Research: run the desk on a ticker, watch it work, read and question the report."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from equitydesk.graph.qa import ReportQA
from equitydesk.policy.guards import InputError
from equitydesk.schemas import RATING_LABEL
from equitydesk.tools.evidence import CATEGORIES, Fact
from equitydesk.ui.common import (ANALYST_TITLE, NODE_LABEL, RATING_COLOR, STANCE_COLOR, desk, md, mode, money,
                                  price_chart)

d = desk()
s = d.s

# -- a report opened from another page -----------------------------------------------------------
if run_id := st.session_state.pop("open_run", None):
    rep = d.store.get_report(run_id)
    if rep:
        st.session_state["report"] = rep
    else:
        st.warning(f"No saved report for run {run_id}.")

st.title("Equity Research Desk")
st.caption("Four analysts in parallel → bull/bear debate → portfolio manager → code-run grounding check. "
           "Every number is computed in code and every claim cites its evidence.")

# -- the request form ----------------------------------------------------------------------------
prefill = st.session_state.pop("prefill_ticker", None)
with st.form("request", border=True):
    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1])
    if mode() == "demo":
        options = [t for t in d.provider.available() if t != s.data.benchmark]
        idx = options.index(prefill) if prefill in options else 0
        ticker = c1.selectbox("Ticker (recorded data)", options, index=idx)
    else:
        ticker = c1.text_input("Ticker", value=prefill or st.session_state.get("last_ticker", "NVDA"),
                               help="Yahoo Finance symbol: AAPL, BRK-B, 7203.T, ...")
    rounds = c2.slider("Debate rounds", 1, s.graph.max_debate_rounds, s.graph.debate_rounds)
    language = c3.selectbox("Report language", ["English", "Chinese"],
                            index=["English", "Chinese"].index(s.graph.language))
    c4.write("")
    submitted = c4.form_submit_button("Run analysis", type="primary", icon=":material/play_arrow:", width="stretch")


def drive(events, total_steps: int, title: str) -> None:
    """Consume a run's event stream into a progress bar and a live status log."""
    bar = st.progress(0.0, text="Starting…")
    done = 0
    with st.status(title, expanded=True) as status:
        for ev in events:
            if ev["type"] == "node":
                done += 1
                node, delta, u = ev["node"], ev["delta"], ev["usage"]
                detail = ""
                if node == "collect_data":
                    detail = f" · {len(delta.get('facts', []))} facts"
                elif node in ANALYST_TITLE:
                    r = (delta.get("analyst_reports") or {}).get(node, {})
                    detail = " · FAILED" if r.get("failed") else f" · {r.get('stance')} ({r.get('score', 0):+.2f})"
                elif node in ("bull", "bear"):
                    detail = f" · round {(delta.get('debate') or [{}])[0].get('round')}"
                elif node == "grounding":
                    g = delta.get("grounding", {})
                    detail = " · passed" if g.get("passed") else (" · failed → PM revises" if g.get("retry") else " · failed")
                elif node == "pm":
                    detail = f" · {(delta.get('decision') or {}).get('rating')}"
                status.write(f":material/check_circle: {NODE_LABEL.get(node, node)}{detail}")
                bar.progress(min(done / total_steps, 1.0),
                             text=f"{NODE_LABEL.get(node, node)} · {u['tokens']:,} tokens · ${u['cost_usd']:.4f}")
            elif ev["type"] == "error":
                status.update(label=f"Run stopped: {ev['kind']}", state="error")
                st.session_state["failed_run"] = ev
                bar.empty()
            else:
                status.update(label="Analysis complete", state="complete", expanded=False)
                st.session_state["report"] = ev["report"]
                st.session_state.pop("failed_run", None)
                bar.empty()


if submitted:
    try:
        rid = d.start(ticker, rounds, language)
    except InputError as e:
        st.error(str(e))
    else:
        st.session_state["last_ticker"] = ticker.strip().upper()
        drive(d.stream(rid), 2 + 4 + 2 * d.s.clamp_rounds(rounds) + 3, f"Analysing {ticker.upper()}…")

if fail := st.session_state.get("failed_run"):
    with st.container(border=True):
        st.error(f"**{fail['kind']}** — {fail['error']}")
        st.caption("Every finished step is checkpointed. Fix the cause (key, budget, network) and resume: "
                   "completed steps are not re-run or re-billed.")
        if st.button("Resume run", icon=":material/replay:", type="primary"):
            remaining = len(d.pending_nodes(fail["run_id"])) + 6
            drive(d.stream(fail["run_id"], resume=True), remaining, f"Resuming {fail['run_id']}…")
            st.rerun()

rep = st.session_state.get("report")
if not rep:
    st.info("Pick a ticker and run the desk. In **Offline demo** mode it runs on recorded data with a scripted "
            "model at no cost; in **Live** mode it uses Yahoo Finance and the model in settings.yaml.")
    st.stop()

# -- the report ----------------------------------------------------------------------------------
company = rep.get("company") or {}
st.divider()
st.subheader(f"{company.get('name', rep['ticker'])} ({rep['ticker']})")
st.caption(" · ".join(x for x in [company.get("sector"), company.get("industry"), f"data as of {rep.get('as_of')}",
                                  f"source {rep.get('data_source')}", f"run {rep['run_id']}"] if x))

if rep.get("status") != "ok":
    for w in rep.get("warnings", []):
        st.warning(w)
    st.stop()

dec, plan, g = rep["decision"], rep["plan"], rep.get("grounding") or {}
facts = {f["id"]: Fact(**f) for f in rep.get("facts", [])}
m = st.columns(6)
with m[0]:
    st.caption("Rating")
    st.badge(RATING_LABEL[dec["rating"]], color=RATING_COLOR[dec["rating"]])
m[1].metric("Conviction", f"{dec['conviction']}/100")
m[2].metric("Horizon", f"{dec['horizon_months']} mo")
m[3].metric("Last close", money(rep.get("price")),
            delta=facts["T.ret_1m"].display() + " 1m" if "T.ret_1m" in facts else None)
m[4].metric("Grounding", "Passed" if g.get("passed") else "Failed",
            delta=f"{g.get('citation_coverage', 0):.0%} cited", delta_color="off")
u = rep.get("usage") or {}
m[5].metric("Run cost", f"${u.get('cost_usd', 0):.3f}", delta=f"{u.get('tokens', 0):,} tokens", delta_color="off")

notes = rep.get("consistency", []) + rep.get("warnings", [])
if notes:
    with st.expander(f"Quality warnings ({len(notes)})", icon=":material/warning:"):
        for w in notes:
            st.markdown(f"- {md(w)}")

tabs = st.tabs(["Summary", "Chart", "Analysts", "Debate", "Evidence", "Ask the desk", "Report"])

with tabs[0]:
    st.markdown("#### Thesis")
    st.markdown(md(dec["thesis"]))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Drivers")
        for p in dec.get("drivers", []):
            st.markdown(f"- {md(p['claim'])}")
    with c2:
        st.markdown("#### Risks")
        for p in dec.get("risks", []):
            st.markdown(f"- {md(p['claim'])}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Catalysts")
        for c in dec.get("catalysts") or ["None given"]:
            st.markdown(f"- {md(c)}")
    with c2:
        st.markdown("#### What would make this wrong")
        for c in dec.get("invalidation") or ["None given"]:
            st.markdown(f"- {md(c)}")
    with st.container(border=True):
        st.markdown("#### Trade plan · computed in code from your risk profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Protective stop", money(plan.get("stop")),
                  delta=f"-{plan['stop_pct']:.1f}%" if plan.get("stop_pct") else None)
        p2.metric("Position size", f"{plan['size_pct']:.1f}%" if plan.get("size_pct") is not None else "—")
        p3.metric("Shares", f"{plan['shares']:,}" if plan.get("shares") is not None else "—")
        p4.metric("At risk if stopped", money(plan.get("risk_usd")) if plan.get("risk_usd") is not None else "—")
        st.markdown(f"**Action:** {md(plan['action'])}")
        st.caption(plan["rule"])
    if dec.get("memory_note"):
        st.markdown("#### Memory")
        st.markdown(md(dec["memory_note"]))
    if dec.get("override_rationale"):
        st.markdown("#### Override rationale")
        st.markdown(md(dec["override_rationale"]))

with tabs[1]:
    try:
        hist = d.provider.history(rep["ticker"], s.data.history_period)
        if hist.empty:
            st.warning("No price history available for the chart.")
        else:
            st.plotly_chart(price_chart(hist, plan), key="price_chart")
    except Exception as e:  # the report stays readable when the network is down
        st.warning(f"Chart unavailable: {e}")

with tabs[2]:
    cols = st.columns(2)
    for i, (name, a) in enumerate((rep.get("analysts") or {}).items()):
        with cols[i % 2].container(border=True):
            st.markdown(f"#### {ANALYST_TITLE.get(name, name)}")
            if a.get("failed"):
                st.error(a.get("error", "failed"))
                continue
            st.badge(f"{a['stance']} {a['score']:+.2f}", color=STANCE_COLOR[a["stance"]])
            st.caption(f"Confidence {a['confidence']}/100 · grounding "
                       f"{'passed' if (a.get('grounding') or {}).get('passed') else 'issues'} · "
                       f"{(a.get('grounding') or {}).get('citation_coverage', 0):.0%} cited")
            st.markdown(md(a["summary"]))
            for p in a.get("points", []):
                st.markdown(f"- {md(p['claim'])}")
            if a.get("data_gaps"):
                st.caption("Data gaps: " + "; ".join(a["data_gaps"]))

with tabs[3]:
    for t in rep.get("debate", []):
        with st.chat_message("user" if t["side"] == "bull" else "assistant",
                             avatar=":material/trending_up:" if t["side"] == "bull" else ":material/trending_down:"):
            st.markdown(f"**Round {t['round']} · {t['side'].upper()}**")
            st.markdown(md(t["argument"]))
            if t.get("rebuttal"):
                st.caption("Rebuttal: " + t["rebuttal"].replace("$", "\\$"))

with tabs[4]:
    df = pd.DataFrame([{"id": f.id, "category": CATEGORIES.get(f.id.split(".")[0], ""), "item": f.label,
                        "value": f.display(), "as of": f.as_of, "source": f.source, "flagged": f.flagged,
                        "url": f.url or None} for f in facts.values()])
    cat = st.segmented_control("Category", sorted(df["category"].unique()), selection_mode="multi",
                               default=sorted(df["category"].unique()))
    st.dataframe(df[df["category"].isin(cat or [])], hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("link", display_text="open"),
                                "flagged": st.column_config.CheckboxColumn("injection flag")})

with tabs[5]:
    qa = ReportQA(d)
    st.caption("Ask follow-up questions. Answers come from this report's evidence only; the conversation is "
               "checkpointed per report and survives a reload.")
    try:
        history = qa.history(rep["run_id"], rep)
    except Exception:
        history = []
    for role, text in history:
        with st.chat_message(role):
            st.markdown(md(text))
    if q := st.chat_input("e.g. Why is the stop where it is? How does valuation compare with growth?"):
        with st.chat_message("user"):
            st.markdown(md(q))
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                st.markdown(md(qa.ask(rep["run_id"], rep, q)))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

with tabs[6]:
    c1, c2 = st.columns(2)
    c1.download_button("Download Markdown", rep["markdown"], file_name=f"{rep['run_id']}.md", mime="text/markdown",
                       icon=":material/download:", width="stretch")
    c2.download_button("Download JSON", json.dumps(rep, indent=1, default=str), file_name=f"{rep['run_id']}.json",
                       mime="application/json", icon=":material/data_object:", width="stretch")
    st.markdown(md(rep["markdown"]))
