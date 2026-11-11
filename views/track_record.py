"""Track record: every past call scored against what the stock then did versus the benchmark.
This is the desk's evaluation in production."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from equitydesk.memory.track import summary, track_record
from equitydesk.ui.common import desk, open_report

d = desk()
s = d.s
st.title("Track record")
st.caption(f"Each call is scored on excess return versus {s.data.benchmark} since the day it was made. "
           f"BUY: hit if it beat the benchmark. SELL: hit if it lagged. HOLD: hit if within "
           f"±{s.grounding.hold_band_pct:g}%. Calls younger than a few days are mostly noise.")

calls = d.store.list_analyses(limit=1000)
if not calls:
    st.info("No calls yet. Run the desk on a ticker first.")
    st.stop()

min_days = st.slider("Only score calls at least this many days old", 0, 90, s.grounding.min_days_to_score)
prices, bench = {}, {}
with st.spinner("Fetching current prices…"):
    for t in {c["ticker"] for c in calls} | {s.data.benchmark}:
        try:
            h = d.provider.history(t, "5d")
            if not h.empty:
                prices[t] = float(h["Close"].iloc[-1])
        except Exception:
            pass
bench[s.data.benchmark] = prices.get(s.data.benchmark)
df = track_record(calls, prices, bench, s.grounding.hold_band_pct, min_days)
summ = summary(df)

c = st.columns(4)
c[0].metric("Calls", summ["calls"])
c[1].metric("Hit rate", f"{summ['hit_rate']:.0%}" if summ["hit_rate"] is not None else "—")
c[2].metric("Avg excess, buys", f"{summ['avg_excess_buys']:+.1f}%" if summ["avg_excess_buys"] is not None else "—")
c[3].metric("Avg excess, sells", f"{summ['avg_excess_sells']:+.1f}%" if summ["avg_excess_sells"] is not None else "—",
            help="Negative is good for a sell call.")

if not df.empty and df["excess_pct"].notna().any():
    fig = px.scatter(df.dropna(subset=["excess_pct"]), x="created_at", y="excess_pct", color="rating",
                     size=df.dropna(subset=["excess_pct"])["conviction"].fillna(50).clip(lower=10),
                     hover_data=["ticker", "price", "price_now", "verdict"],
                     color_discrete_map={"STRONG_BUY": "#15803d", "BUY": "#22c55e", "HOLD": "#94a3b8",
                                         "SELL": "#f87171", "STRONG_SELL": "#b91c1c"},
                     labels={"excess_pct": f"Excess return vs {s.data.benchmark} (%)", "created_at": "Call date"})
    fig.add_hline(y=0, line=dict(color="#64748b", width=1))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, key="track_scatter")

event = st.dataframe(
    df[["created_at", "ticker", "rating", "conviction", "price", "price_now", "return_pct", "bench_pct", "excess_pct",
        "verdict", "days", "grounded", "run_id"]],
    hide_index=True, on_select="rerun", selection_mode="single-row", key="track",
    column_config={"price": st.column_config.NumberColumn("price then", format="$%.2f"),
                   "price_now": st.column_config.NumberColumn(format="$%.2f"),
                   "return_pct": st.column_config.NumberColumn("return", format="%+.1f%%"),
                   "bench_pct": st.column_config.NumberColumn("benchmark", format="%+.1f%%"),
                   "excess_pct": st.column_config.NumberColumn("excess", format="%+.1f%%"),
                   "grounded": st.column_config.CheckboxColumn()})
if event and event.selection and event.selection.rows:
    rid = df.iloc[event.selection.rows[0]]["run_id"]
    if st.button("Open this report", type="primary", icon=":material/open_in_new:"):
        open_report(rid)
