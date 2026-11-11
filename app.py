"""EquityDesk: streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from equitydesk.ui import components as ui
from equitydesk.ui.common import MODES, desk, model_status, reload_settings

st.set_page_config(page_title="EquityDesk", page_icon=":material/query_stats:", layout="wide",
                   initial_sidebar_state="expanded")
ui.style()

with st.sidebar:
    st.markdown("### :material/query_stats: EquityDesk")
    st.caption("Multi-agent equity research · LangGraph + Streamlit")
    st.radio("Mode", list(MODES), format_func=lambda m: MODES[m], key="mode",
             help="Offline demo uses its own data folder (data/demo), so it never touches your real memory.")
    d = desk()
    ok, label = model_status(d)
    (st.success if ok else st.error)(f"Model: {label}", icon=":material/smart_toy:")
    st.caption(f"Data: {d.provider.name} · benchmark {d.s.data.benchmark}  \n"
               f"Budget per run: {d.s.budget.run_token_ceiling:,} tokens / ${d.s.budget.run_cost_ceiling_usd:.2f}  \n"
               f"Web research: {'on' if d.s.research.web_search else 'off'} ({d.s.research.context_size} context)")
    if not ok:
        st.caption("Add the key to `.env` (see `.env.example`), then press Reload — or switch to the offline demo.")
    if st.button("Reload settings", icon=":material/refresh:", help="Re-read settings.yaml, pricing.yaml and .env"):
        reload_settings()
        st.rerun()

pages = [
    st.Page("views/research.py", title="Research", icon=":material/analytics:", default=True),
    st.Page("views/news.py", title="News", icon=":material/newspaper:"),
    st.Page("views/watchlist.py", title="Watchlist", icon=":material/list_alt:"),
    st.Page("views/track_record.py", title="Track record", icon=":material/fact_check:"),
    st.Page("views/memory.py", title="Memory & profile", icon=":material/psychology:"),
    st.Page("views/runs.py", title="Runs & traces", icon=":material/timeline:"),
]
st.navigation(pages, position="top").run()
