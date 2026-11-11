"""Memory & profile: what the desk remembers about you and about its own past calls. All of it editable."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from equitydesk.memory.store import GLOBAL
from equitydesk.policy.guards import InputError, normalize_ticker
from equitydesk.ui import components as ui
from equitydesk.ui.common import desk, open_report

d = desk()
ui.page_header("Memory & profile", "What the desk remembers about you and about its own past calls. All of it editable.")

st.subheader("Risk profile")
st.caption("Used in code for position sizing and stops, and shown to the portfolio manager.")
p = d.store.get_profile()
with st.form("profile"):
    c = st.columns(5)
    pv = c[0].number_input("Portfolio value ($)", min_value=1_000.0, value=float(p["portfolio_value"]), step=10_000.0)
    rpt = c[1].number_input("Risk per trade (%)", 0.1, 5.0, float(p["risk_per_trade_pct"]), 0.1,
                            help="Loss if the protective stop is hit, as % of the portfolio.")
    mpp = c[2].number_input("Max position (%)", 1.0, 100.0, float(p["max_position_pct"]), 1.0)
    tol = c[3].selectbox("Risk tolerance", ["conservative", "moderate", "aggressive"],
                         index=["conservative", "moderate", "aggressive"].index(p["risk_tolerance"]))
    hz = c[4].number_input("Horizon (months)", 1, 24, int(p["horizon_months"]))
    if st.form_submit_button("Save profile", type="primary"):
        d.store.set_profile({"portfolio_value": pv, "risk_per_trade_pct": rpt, "max_position_pct": mpp,
                             "risk_tolerance": tol, "horizon_months": int(hz)})
        st.success("Saved. It applies from the next run.")

st.subheader("Your notes")
st.caption("Things the desk should know: positions you hold, constraints, views. A note on `*` applies to every "
           "ticker. The PM reads them before every decision on that ticker.")
with st.form("note", clear_on_submit=True):
    c1, c2 = st.columns([1, 4])
    nt = c1.text_input("Ticker (or *)", value="*")
    text = c2.text_input("Note", placeholder="e.g. I hold 200 NVDA at $95; never more than 8% in one name.")
    if st.form_submit_button("Add note"):
        try:
            d.store.add_note(GLOBAL if nt.strip() == GLOBAL else normalize_ticker(nt), text)
            st.rerun()
        except (InputError, ValueError) as e:
            st.error(str(e))
for n in d.store.list_notes():
    c1, c2, c3 = st.columns([1, 6, 1])
    c1.code(n["ticker"])
    c2.markdown(n["text"].replace("$", "\\$") + f"  \n:gray[{n['created_at']}]")
    if c3.button("Forget", key=f"del-{n['id']}", icon=":material/delete:"):
        d.store.deactivate_note(n["id"])
        st.rerun()

st.subheader("Past calls (episodic memory)")
st.caption("The PM sees its last three calls on a ticker, scored against what happened since. Forgetting a call "
           "removes it from memory and from the track record.")
calls = d.store.list_analyses(limit=500)
if not calls:
    st.info("No calls yet.")
else:
    df = pd.DataFrame(calls)[["created_at", "ticker", "rating", "conviction", "price", "grounded", "thesis", "id", "run_id"]]
    ev = st.dataframe(df.drop(columns=["id"]), hide_index=True, on_select="rerun", selection_mode="single-row",
                      key="calls", column_config={"price": st.column_config.NumberColumn(format="$%.2f"),
                                                  "grounded": st.column_config.CheckboxColumn(),
                                                  "thesis": st.column_config.TextColumn(width="large")})
    if ev and ev.selection and ev.selection.rows:
        row = df.iloc[ev.selection.rows[0]]
        c1, c2 = st.columns(2)
        if c1.button("Open report", icon=":material/open_in_new:", width="stretch"):
            open_report(row["run_id"])
        if c2.button("Forget this call", icon=":material/delete:", width="stretch"):
            d.store.delete_analysis(row["id"])
            st.rerun()
