"""News tracker: headlines and sell-side actions across the watchlist, with what is new since you last looked."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from equitydesk.tools import news as newslib
from equitydesk.tools.evidence import _action_text
from equitydesk.ui import components as ui
from equitydesk.ui.common import desk

d = desk()
s = d.s
ui.page_header("News tracker", "Headlines and analyst actions across your watchlist. New items are marked; "
                               "headlines not about the company are filtered out in code.")

profile = d.store.get_profile()
tickers = profile["watchlist"]
c1, c2, c3 = st.columns([3, 1.2, 1.2])
chosen = c1.multiselect("Tickers", tickers, default=tickers, help="Edit the list on the Watchlist page.")
only_about = c2.toggle("Only about the company", value=True)
fresh_hours = c3.selectbox("Mark as NEW if first seen within", [6, 24, 72], index=1, format_func=lambda h: f"{h} hours")

with st.spinner("Collecting headlines…"):
    rows, errors = newslib.collect(d.provider, d.store, chosen or [])
for t, e in errors.items():
    st.warning(f"{t}: news unavailable ({e})")

now = datetime.now()


def is_new(r) -> bool:
    try:
        return now - datetime.fromisoformat(r["first_seen"]) <= timedelta(hours=fresh_hours)
    except (KeyError, ValueError):
        return False


shown = [r for r in rows if (r.get("about") or not only_about) and r.get("relevant") != 0]
shown.sort(key=lambda r: (r.get("published") or "", r.get("first_seen") or ""), reverse=True)

# -- sell-side actions across the watchlist (structured, free) -----------------------------------
actions = []
for t in chosen or []:
    try:
        for a in (d.provider.street(t) or {}).get("actions") or []:
            actions.append({**a, "ticker": t})
    except Exception:
        continue
recent = [a for a in actions if a.get("date") and (date.today() - date.fromisoformat(a["date"])).days <= 7]

k = st.columns(4)
k[0].metric("Headlines", len(shown))
k[1].metric("New", sum(1 for r in shown if is_new(r)))
k[2].metric("High materiality", sum(1 for r in shown if r.get("materiality") == "high"))
k[3].metric("Analyst actions (7d)", len(recent),
            delta=f"{sum(1 for a in recent if a['action'] == 'down')} downgrades" if recent else None,
            delta_color="inverse")

tab_h, tab_a = st.tabs(["Headlines", "Analyst actions"])
with tab_h:
    untriaged = [r for r in shown if not r.get("triaged_at")]
    b1, b2 = st.columns([1, 3])
    if b1.button(f"AI triage ({len(untriaged)})", icon=":material/auto_awesome:", disabled=not untriaged,
                 help="One model call classifies tone and materiality for the untriaged headlines (about $0.01)."):
        with st.spinner("Triaging…"):
            try:
                out = newslib.triage(s, d.store, untriaged, backend_factory=d.factory)
                st.toast(f"Triaged {out['triaged']} headlines for ${out['cost_usd']:.4f}")
                st.rerun()
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    b2.caption("Tone and materiality come from the AI triage; whether a headline is about the company is decided "
               "in code from the ticker and company name.")
    out = []
    for r in shown:
        chips = ui.chip(r["ticker"], "blue")
        if is_new(r):
            chips += ui.chip("NEW", "new")
        if r.get("tone"):
            chips += ui.chip(r["tone"], ui.TONE.get(r["tone"], "slate"))
        if r.get("materiality"):
            chips += ui.chip(f"{r['materiality']} materiality", ui.MATERIALITY.get(r["materiality"], "slate"))
        if not r.get("about"):
            chips += ui.chip("not about the company", "slate")
        if r.get("flagged"):
            chips += ui.chip("injection attempt flagged", "red")
        sub = ui.esc(r.get("publisher") or "") + (f" · {ui.esc(r['why'])}" if r.get("why") else "")
        out.append(ui.news_row(r.get("published") or "", r["title"], r.get("url") or "", sub, chips))
    ui.html_block('<div class="ed-card">' + ("".join(out) or "<p class='ed-muted'>No headlines.</p>") + "</div>")

with tab_a:
    if not actions:
        st.info("No sell-side actions available for these tickers.")
    else:
        df = pd.DataFrame(actions).sort_values("date", ascending=False).head(150)
        rows = []
        for a in df.to_dict("records"):
            label, tone = ui.action_tone(_action_text(a))
            rows.append(ui.news_row(a["date"], f"{a['ticker']} · {_action_text(a)}", "", "",
                                    ui.chip(a["ticker"], "blue") + ui.chip(label, tone)))
        ui.html_block('<div class="ed-card">' + "".join(rows) + "</div>")
