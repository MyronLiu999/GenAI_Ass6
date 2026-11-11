"""Research: run the desk on a ticker, watch it work, read and question the report."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from equitydesk.graph.qa import ReportQA
from equitydesk.policy.guards import InputError
from equitydesk.tools.evidence import CATEGORIES, Fact
from equitydesk.ui import components as ui
from equitydesk.ui.common import ANALYST_TITLE, NODE_LABEL, desk, md, mode, money, price_chart

d = desk()
s = d.s

# -- a report opened from another page -----------------------------------------------------------
if run_id := st.session_state.pop("open_run", None):
    rep = d.store.get_report(run_id)
    if rep:
        st.session_state["report"] = rep
    else:
        st.warning(f"No saved report for run {run_id}.")

# -- the request form ----------------------------------------------------------------------------
prefill = st.session_state.pop("prefill_ticker", None)
with st.form("request", border=True):
    c1, c2, c3, c4, c5 = st.columns([2.1, 1.3, 1.1, 1.2, 1])
    if mode() == "demo":
        options = [t for t in d.provider.available() if t != s.data.benchmark]
        idx = options.index(prefill) if prefill in options else 0
        ticker = c1.selectbox("Ticker (recorded data)", options, index=idx)
    else:
        ticker = c1.text_input("Ticker", value=prefill or st.session_state.get("last_ticker", "NVDA"),
                               help="Yahoo Finance symbol: AAPL, BRK-B, 7203.T, ...")
    rounds = c2.slider("Debate rounds", 1, s.graph.max_debate_rounds, s.graph.debate_rounds)
    language = c3.selectbox("Language", ["English", "Chinese"], index=["English", "Chinese"].index(s.graph.language))
    web = c4.toggle("Web research", value=s.research.web_search,
                    help="Search the web for recent sell-side notes and independent research (adds about $0.10).")
    c5.write("")
    submitted = c5.form_submit_button("Run analysis", type="primary", icon=":material/play_arrow:", width="stretch")


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
                elif node == "web_research":
                    r = delta.get("research") or {}
                    detail = (f" · {len(r.get('notes', []))} notes from {len(r.get('sources', []))} cited sources"
                              + (f", {r['dropped']} dropped" if r.get("dropped") else "")) if r.get("status") == "ok" \
                        else f" · {r.get('status')}"
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
        rid = d.start(ticker, rounds, language, web)
    except InputError as e:
        st.error(str(e))
    else:
        st.session_state["last_ticker"] = ticker.strip().upper()
        drive(d.stream(rid), 2 + 6 + 2 * d.s.clamp_rounds(rounds) + 3, f"Analysing {ticker.upper()}…")

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
    ui.html_block("""
<div class="ed-hero">
  <div class="t">Equity Research Desk</div>
  <div class="s">Five analysts in parallel, web research with verified sources, a bull/bear debate and a portfolio
  manager, with every number computed in code and every claim fact-checked before you see it.</div>
</div>""")
    cols = st.columns(4)
    for c, (n, h, t) in zip(cols, [
        ("01", "Evidence pack", "80+ figures computed in code: technicals, risk, fundamentals, sell-side actions and "
                                "estimate revisions. Every one has an id agents must cite."),
        ("02", "Street research", "Recent analyst notes and independent research found on the web. Only sources the "
                                  "search tool actually cited survive."),
        ("03", "Debate and decision", "Bull and bear argue; the PM states the key debates and its variant view "
                                      "against the street, then a code check verifies citations and numbers."),
        ("04", "Memory", "Every call is stored with its price and scored later against the benchmark; the PM sees "
                         "its own track record before deciding."),
    ]):
        c.markdown(f'<div class="ed-feature"><div class="n">{n}</div><div class="h">{h}</div>'
                   f'<div class="d">{t}</div></div>', unsafe_allow_html=True)
    st.caption("Offline demo mode (sidebar) runs on recorded data with a scripted model at no cost.")
    st.stop()

# -- the report ----------------------------------------------------------------------------------
if rep.get("status") != "ok":
    st.subheader(f"{rep['ticker']}: no report")
    for w in rep.get("warnings", []):
        st.warning(w)
    st.stop()

dec, plan, g = rep["decision"], rep["plan"], rep.get("grounding") or {}
facts = {f["id"]: Fact(**f) for f in rep.get("facts", [])}
analysts = rep.get("analysts") or {}
research = rep.get("research") or {}
ui.hero(rep, facts)

left, right = st.columns([1.35, 1])
with left:
    ui.card("Thesis", f"<p>{ui.rich(dec['thesis'])}</p>", accent="")
    if dec.get("variant_view"):
        ui.card("Variant view · where we differ from the street", f"<p>{ui.rich(dec['variant_view'])}</p>",
                accent="amber")
with right:
    ui.section("Desk scorecard")
    st.plotly_chart(ui.scorecard(analysts, dec, ANALYST_TITLE), key="scorecard", config={"displayModeBar": False})

notes = rep.get("consistency", []) + rep.get("warnings", [])
if notes:
    with st.expander(f"Quality warnings ({len(notes)})", icon=":material/warning:"):
        for w in notes:
            st.markdown(f"- {md(w)}")

tabs = st.tabs(["Summary", "Street research", f"News ({len((analysts.get('sentiment') or {}).get('headlines') or [])})",
                "Chart", "Analysts", "Debate", "Evidence", "Ask the desk", "Report"])

with tabs[0]:
    if dec.get("key_debates"):
        ui.card("Key debates", "<ul>" + "".join(f"<li>{ui.rich(x)}</li>" for x in dec["key_debates"]) + "</ul>",
                accent="slate")
    c1, c2 = st.columns(2)
    with c1:
        ui.card("Drivers", ui.point_list(dec.get("drivers")), accent="green")
        ui.card("Catalysts", "<ul>" + "".join(f"<li>{ui.rich(x)}</li>" for x in dec.get("catalysts") or ["None given"])
                + "</ul>")
    with c2:
        ui.card("Risks", ui.point_list(dec.get("risks")), accent="red")
        ui.card("What would make this call wrong",
                "<ul>" + "".join(f"<li>{ui.rich(x)}</li>" for x in dec.get("invalidation") or []) + "</ul>")
    ui.section("Trade plan · computed in code from your risk profile")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Protective stop", money(plan.get("stop")),
              delta=f"-{plan['stop_pct']:.1f}%" if plan.get("stop_pct") else None)
    p2.metric("Position size", f"{plan['size_pct']:.1f}%" if plan.get("size_pct") is not None else "—")
    p3.metric("Shares", f"{plan['shares']:,}" if plan.get("shares") is not None else "—")
    p4.metric("At risk if stopped", money(plan.get("risk_usd")) if plan.get("risk_usd") is not None else "—")
    ui.card("Action", f"<p>{ui.esc(plan['action'])}</p><div class='meta'>{ui.esc(plan['rule'])}</div>")
    if dec.get("memory_note"):
        ui.card("Memory", f"<p>{ui.rich(dec['memory_note'])}</p>", accent="slate")
    if dec.get("override_rationale"):
        ui.card("Override rationale", f"<p>{ui.rich(dec['override_rationale'])}</p>", accent="amber")

with tabs[1]:
    street = analysts.get("street") or {}
    get = lambda k: facts[k].numeric if k in facts else None  # noqa: E731
    k = st.columns(5)
    k[0].metric("Upgrades / downgrades (90d)",
                f"{int(get('A.upgrades_90d') or 0)} / {int(get('A.downgrades_90d') or 0)}")
    k[1].metric("Target raises / cuts (30d)",
                f"{int(get('A.target_raises_30d') or 0)} / {int(get('A.target_cuts_30d') or 0)}")
    k[2].metric("Fresh 30-day median target", money(get("A.fresh_target_median")),
                delta=ui.signed(get("A.fresh_target_upside")) if get("A.fresh_target_upside") is not None else None)
    k[3].metric("Mean target (all)", money(get("S.target_mean")),
                delta=ui.signed(get("S.target_upside")) if get("S.target_upside") is not None else None)
    k[4].metric("Next-year EPS revision (90d)", ui.signed(get("A.eps_ny_rev_90d")))
    if street and not street.get("failed"):
        ui.card("The street's view", f"<p>{ui.rich(street.get('consensus_view', ''))}</p>"
                + (f"<div class='meta'>Street analyst: {ui.esc(street['stance'])} {street['score']:+.2f} · "
                   f"confidence {street['confidence']}</div>"), accent="")
        c1, c2 = st.columns(2)
        with c1:
            ui.card("Street bull case", ui.point_list(street.get("street_bull")), accent="green")
        with c2:
            ui.card("Street bear case", ui.point_list(street.get("street_bear")), accent="red")
        if street.get("recent_changes"):
            ui.card("What changed in the last 30 days", ui.point_list(street["recent_changes"]), accent="amber")
        if street.get("key_debates"):
            ui.card("Where the street is divided",
                    "<ul>" + "".join(f"<li>{ui.rich(x)}</li>" for x in street["key_debates"]) + "</ul>", accent="slate")
    actions = [f for fid, f in facts.items() if fid.startswith("A.action_")]
    if actions:
        ui.section("Latest sell-side actions")
        rows = []
        for f in actions:
            label, tone = ui.action_tone(f.value)
            rows.append(ui.news_row(f.as_of, f.value, "", f'<span class="ed-cite">{f.id}</span>', ui.chip(label, tone)))
        ui.html_block('<div class="ed-card">' + "".join(rows) + "</div>")
    ui.section(f"Research notes found on the web · {len(research.get('notes', []))} kept with verified sources"
               + (f" · {research['dropped']} dropped" if research.get("dropped") else ""))
    if research.get("notes"):
        cols = st.columns(2)
        for i, n in enumerate(research["notes"]):
            with cols[i % 2]:
                ui.note_card(n)
    else:
        st.info({"off": "Web research was off for this run.",
                 "unsupported": "Web research needs the OpenAI provider; this run used the structured sell-side data only."}
                .get(research.get("status"), "No research notes for this run."))

with tabs[2]:
    sent = analysts.get("sentiment") or {}
    heads = {h["id"]: h for h in sent.get("headlines") or []}
    news = [f for fid, f in facts.items() if fid.startswith("N.")]
    if sent and not sent.get("failed"):
        ui.card("News read", f"<p>{ui.rich(sent.get('summary', ''))}</p>", accent="")
    only = st.toggle("Only headlines about this company", value=True)
    rows = []
    order = {"high": 0, "medium": 1, "low": 2}
    for f in sorted(news, key=lambda f: (order.get(heads.get(f.id, {}).get("materiality"), 3), f.as_of), reverse=False):
        h = heads.get(f.id, {})
        if only and h and not h.get("relevant", True):
            continue
        chips = ""
        if h:
            chips += ui.chip(h.get("tone", ""), ui.TONE.get(h.get("tone"), "slate"))
            chips += ui.chip(f"{h.get('materiality', '')} materiality", ui.MATERIALITY.get(h.get("materiality"), "slate"))
            if not h.get("relevant", True):
                chips += ui.chip("not about this company", "slate")
        if f.flagged:
            chips += ui.chip("injection attempt flagged", "red")
        rows.append(ui.news_row(f.as_of, f.label, f.url, f"{ui.esc(f.value)} · {ui.esc(h.get('why', ''))} "
                                f'<span class="ed-cite">{f.id}</span>', chips))
    ui.html_block('<div class="ed-card">' + ("".join(rows) or "<p class='ed-muted'>No headlines.</p>") + "</div>")
    st.caption("Track headlines across your whole watchlist on the News page.")

with tabs[3]:
    try:
        hist = d.provider.history(rep["ticker"], s.data.history_period)
        if hist.empty:
            st.warning("No price history available for the chart.")
        else:
            st.plotly_chart(price_chart(hist, plan), key="price_chart")
    except Exception as e:  # the report stays readable when the network is down
        st.warning(f"Chart unavailable: {e}")

with tabs[4]:
    cols = st.columns(2)
    for i, (name, a) in enumerate(analysts.items()):
        with cols[i % 2]:
            if a.get("failed"):
                ui.card(ANALYST_TITLE.get(name, name), f"<p>{ui.esc(a.get('error', 'failed'))}</p>", accent="red")
                continue
            tone = ui.TONE.get(a["stance"], "slate")
            gr = a.get("grounding") or {}
            head = (ui.chip(f"{a['stance']} {a['score']:+.2f}", tone) + ui.chip(f"confidence {a['confidence']}", "blue")
                    + ui.chip(f"{gr.get('citation_coverage', 0):.0%} cited", "violet"))
            gaps = f"<div class='meta'>Data gaps: {ui.esc('; '.join(a['data_gaps']))}</div>" if a.get("data_gaps") else ""
            ui.card(ANALYST_TITLE.get(name, name), f"<div>{head}</div><p>{ui.rich(a['summary'])}</p>"
                    + ui.point_list(a.get("points")) + gaps, accent={"green": "green", "red": "red"}.get(tone, "slate"))

with tabs[5]:
    for t in rep.get("debate", []):
        with st.chat_message("user" if t["side"] == "bull" else "assistant",
                             avatar=":material/trending_up:" if t["side"] == "bull" else ":material/trending_down:"):
            st.markdown(f"**Round {t['round']} · {t['side'].upper()}**")
            st.markdown(md(t["argument"]))
            if t.get("rebuttal"):
                st.caption("Rebuttal: " + md(t["rebuttal"]))

with tabs[6]:
    df = pd.DataFrame([{"id": f.id, "category": CATEGORIES.get(f.id.split(".")[0], ""), "item": f.label,
                        "value": f.display(), "as of": f.as_of, "source": f.source, "flagged": f.flagged,
                        "url": f.url or None} for f in facts.values()])
    cat = st.segmented_control("Category", sorted(df["category"].unique()), selection_mode="multi",
                               default=sorted(df["category"].unique()))
    st.dataframe(df[df["category"].isin(cat or [])], hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("link", display_text="open"),
                                "flagged": st.column_config.CheckboxColumn("injection flag"),
                                "value": st.column_config.TextColumn(width="large")})

with tabs[7]:
    qa = ReportQA(d)
    st.caption("Ask follow-up questions. Answers come from this report's evidence only, including the research "
               "notes; the conversation is checkpointed per report and survives a reload.")
    try:
        history = qa.history(rep["run_id"], rep)
    except Exception:
        history = []
    for role, text in history:
        with st.chat_message(role):
            st.markdown(md(text))
    if q := st.chat_input("e.g. Why did Baird downgrade? Is the street too optimistic on next year's EPS?"):
        with st.chat_message("user"):
            st.markdown(md(q))
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                st.markdown(md(qa.ask(rep["run_id"], rep, q)))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

with tabs[8]:
    c1, c2 = st.columns(2)
    c1.download_button("Download Markdown", rep["markdown"], file_name=f"{rep['run_id']}.md", mime="text/markdown",
                       icon=":material/download:", width="stretch")
    c2.download_button("Download JSON", json.dumps(rep, indent=1, default=str), file_name=f"{rep['run_id']}.json",
                       mime="application/json", icon=":material/data_object:", width="stretch")
    st.markdown(md(rep["markdown"]))
