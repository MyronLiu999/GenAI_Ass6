"""Shared Streamlit helpers: the cached Desk per mode, labels, Markdown escaping, charts."""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from equitydesk.config import load_env, load_settings, plausible_key
from equitydesk.runner import Desk
from equitydesk.tools import indicators as ind

MODES = {"live": "Live: Yahoo Finance + the model in settings.yaml",
         "demo": "Offline demo: recorded data + scripted model (free)"}
RATING_COLOR = {"STRONG_BUY": "green", "BUY": "green", "HOLD": "gray", "SELL": "red", "STRONG_SELL": "red"}
STANCE_COLOR = {"bullish": "green", "neutral": "gray", "bearish": "red"}
NODE_LABEL = {"collect_data": "Market data & evidence pack", "recall_memory": "Memory: past calls & notes",
              "fundamental": "Fundamental analyst", "technical": "Technical analyst",
              "sentiment": "Sentiment & news analyst", "risk": "Risk manager",
              "web_research": "Web research: street notes + industry drivers, sources verified",
              "street": "Street research analyst", "earnings": "Earnings & demand analyst",
              "bull": "Bull researcher",
              "bear": "Bear researcher", "pm": "Portfolio manager", "grounding": "Grounding check",
              "finalize": "Report & memory write"}
ANALYST_TITLE = {"fundamental": "Fundamental", "technical": "Technical", "sentiment": "Sentiment & News", "risk": "Risk",
                 "street": "Street research", "earnings": "Earnings & demand"}
CITE = re.compile(r"\[\s*([A-Z]\.[A-Za-z0-9_]+(?:\s*[,;]\s*[A-Z]\.[A-Za-z0-9_]+)*)\s*\]")


@st.cache_resource(show_spinner=False)
def _desk(mode: str) -> Desk:
    load_env()
    if mode == "demo":
        s = load_settings(overrides={"model": {"provider": "scripted", "name": "scripted"}, "roles": {},
                                     "data": {"provider": "fixtures"}, "paths": {"data_dir": "data/demo"}})
    else:
        s = load_settings()
    return Desk(s)


def mode() -> str:
    return st.session_state.get("mode", "live")


def desk() -> Desk:
    return _desk(mode())


def reload_settings() -> None:
    _desk.clear()


def model_status(d: Desk) -> tuple[bool, str]:
    cfg = d.s.model_for("pm")
    if cfg.provider == "scripted":
        return True, "scripted (offline)"
    env = cfg.api_key_env or ("ANTHROPIC_API_KEY" if cfg.provider == "anthropic" else "OPENAI_API_KEY")
    ok = plausible_key(os.environ.get(env))
    names = {"PM": cfg.name, "earnings": d.s.model_for("analyst", "earnings").name,
             "analysts": d.s.model_for("analyst").name}
    label = cfg.name if len(set(names.values())) == 1 else " · ".join(f"{k} {v}" for k, v in names.items())
    return ok, f"{cfg.provider} · {label}" + ("" if ok else f" · {env} missing or invalid")


def md(text: str) -> str:
    """Safe for st.markdown: '$' would start LaTeX; citations become inline code."""
    return CITE.sub(r"`\1`", (text or "").replace("$", "\\$"))


def open_report(run_id: str) -> None:
    st.session_state["open_run"] = run_id
    st.switch_page("views/research.py")


def money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def pct(v: float | None, signed: bool = True) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.1f}%" if signed else f"{v:.1f}%")


def price_chart(hist: pd.DataFrame, plan: dict | None = None, title: str = "") -> go.Figure:
    close = hist["Close"]
    bb = ind.bollinger(close)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.16, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=close,
                                 name="Price", showlegend=False), row=1, col=1)
    for n, color in ((50, "#f59e0b"), (200, "#6366f1")):
        fig.add_trace(go.Scatter(x=hist.index, y=ind.sma(close, n), name=f"SMA {n}", line=dict(width=1.4, color=color)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=bb["upper"], name="Bollinger", line=dict(width=0.8, dash="dot", color="#94a3b8")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=bb["lower"], showlegend=False, line=dict(width=0.8, dash="dot", color="#94a3b8")),
                  row=1, col=1)
    for key, label, color in (("stop", "Stop", "#ef4444"), ("support", "60d support", "#22c55e"),
                              ("resistance", "60d resistance", "#a855f7")):
        if plan and plan.get(key):
            fig.add_hline(y=plan[key], line=dict(color=color, width=1, dash="dash"), row=1, col=1,
                          annotation_text=f"{label} {plan[key]:,.2f}", annotation_position="top left")
    up = (hist["Close"] >= hist["Open"]).map({True: "#22c55e", False: "#ef4444"})
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=up, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=ind.rsi(close), name="RSI 14", line=dict(width=1.2, color="#0ea5e9")),
                  row=3, col=1)
    for lvl in (70, 30):
        fig.add_hline(y=lvl, line=dict(color="#94a3b8", width=0.8, dash="dot"), row=3, col=1)
    start = max(hist.index[-1] - pd.Timedelta(days=365), hist.index[0])
    fig.update_xaxes(range=[start, hist.index[-1] + pd.Timedelta(days=3)])
    window = hist[hist.index >= start]
    levels = [plan[k] for k in ("stop", "support", "resistance") if plan and plan.get(k)]
    lo, hi = min([window["Low"].min(), *levels]), max([window["High"].max(), *levels])
    pad = (hi - lo) * 0.04
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    fig.update_yaxes(range=[0, window["Volume"].max() * 1.05], row=2, col=1)
    fig.update_layout(height=680, margin=dict(l=10, r=10, t=40 if title else 10, b=10), title=title or None,
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    return fig
