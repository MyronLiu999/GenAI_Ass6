"""Watchlist: a quantitative snapshot of many tickers at once. Pure code, no model calls, no cost."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from equitydesk.policy.guards import InputError, normalize_ticker
from equitydesk.tools import indicators as ind
from equitydesk.tools import risk as rk
from equitydesk.ui import components as ui
from equitydesk.ui.common import desk

d = desk()
s = d.s
ui.page_header("Watchlist", "Price, momentum, trend, risk and the desk's last call for every name: computed in code, no tokens.")
st.caption("Computed in code from price history (cached); no model is called. Pick a row to send it to the desk.")

profile = d.store.get_profile()
with st.expander("Edit watchlist", icon=":material/edit:"):
    raw = st.text_input("Tickers, comma-separated", ", ".join(profile["watchlist"]))
    if st.button("Save watchlist"):
        try:
            tickers = list(dict.fromkeys(normalize_ticker(t) for t in raw.split(",") if t.strip()))
        except InputError as e:
            st.error(str(e))
        else:
            d.store.set_profile({"watchlist": tickers})
            st.rerun()

tickers = d.store.get_profile()["watchlist"]
if not tickers:
    st.info("The watchlist is empty.")
    st.stop()

last_calls = {}
for c in d.store.list_analyses(limit=1000):
    last_calls.setdefault(c["ticker"], c)

bench = d.provider.history(s.data.benchmark, "1y")
rows = []
with st.spinner("Loading prices…"):
    for t in tickers:
        try:
            h = d.provider.history(t, "1y")
        except Exception as e:
            rows.append({"ticker": t, "error": str(e)[:80]})
            continue
        if h.empty or len(h) < 30:
            rows.append({"ticker": t, "error": "no data"})
            continue
        c = h["Close"]
        r = rk.daily_returns(c)
        sma200 = ind.last(ind.sma(c, 200))
        call = last_calls.get(t)
        rows.append({
            "ticker": t, "price": float(c.iloc[-1]),
            "1d %": ind.pct_return(c, 1), "1m %": ind.pct_return(c, 21), "3m %": ind.pct_return(c, 63),
            "YTD %": ind.ytd_return(c),
            "vs SMA200 %": (float(c.iloc[-1]) / sma200 - 1) * 100 if sma200 else None,
            "RSI 14": ind.last(ind.rsi(c)), "vol 1y %": rk.ann_vol_pct(r),
            "beta": rk.beta(r, rk.daily_returns(bench["Close"])) if not bench.empty else None,
            "from 52w high %": (float(c.iloc[-1]) / float(c.max()) - 1) * 100,
            "trend": [float(x) for x in c.iloc[-63:]],
            "last call": f"{call['rating']} ({call['created_at'][:10]})" if call else "",
        })

df = pd.DataFrame(rows)
fmt_pct = st.column_config.NumberColumn(format="%+.1f%%")
event = st.dataframe(
    df, hide_index=True, on_select="rerun", selection_mode="single-row", key="watch",
    column_config={"price": st.column_config.NumberColumn(format="$%.2f"), "1d %": fmt_pct, "1m %": fmt_pct,
                   "3m %": fmt_pct, "YTD %": fmt_pct, "vs SMA200 %": fmt_pct, "from 52w high %": fmt_pct,
                   "vol 1y %": st.column_config.NumberColumn(format="%.1f%%"),
                   "RSI 14": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                   "beta": st.column_config.NumberColumn(format="%.2f"),
                   "trend": st.column_config.LineChartColumn("3 months")})

sel = event.selection.rows if event and event.selection else []
if sel:
    t = df.iloc[sel[0]]["ticker"]
    if st.button(f"Analyse {t} on the desk", type="primary", icon=":material/analytics:"):
        st.session_state["prefill_ticker"] = t
        st.switch_page("views/research.py")
