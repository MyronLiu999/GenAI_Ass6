"""Visual components shared by every page. Every piece of third-party text (news, research notes, model
output) passes through esc() before it is placed in HTML, and only http(s) links are ever emitted."""

from __future__ import annotations

import html
import re

import plotly.graph_objects as go
import streamlit as st

from equitydesk.policy.provenance import normalize_url
from equitydesk.schemas import RATING_LABEL, RATING_SCORE

CITE = re.compile(r"\[\s*([A-Z]\.[A-Za-z0-9_]+(?:\s*[,;]\s*[A-Z]\.[A-Za-z0-9_]+)*)\s*\]")
GREEN, RED, AMBER, SLATE, BLUE = "#15803d", "#b91c1c", "#b45309", "#64748b", "#1d4ed8"

CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 14px;}
[data-testid="stMetricLabel"] p {font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; color: #64748b;}
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid #e2e8f0;}
.stTabs [data-baseweb="tab"] {padding: 8px 14px; border-radius: 8px 8px 0 0; font-weight: 600;}
.ed-hero {background: linear-gradient(120deg, #0b1f3a 0%, #13336b 55%, #1d4ed8 100%); color: #fff;
  border-radius: 18px; padding: 22px 28px; margin: 2px 0 16px; box-shadow: 0 10px 30px rgba(15, 23, 42, .18);}
.ed-hero .t {font-size: 1.85rem; font-weight: 800; line-height: 1.15; margin: 0;}
.ed-hero .s {color: #c7d2fe; font-size: .88rem; margin-top: 4px;}
.ed-hero .row {display: flex; flex-wrap: wrap; gap: 26px; align-items: flex-end; margin-top: 16px;}
.ed-hero .k {font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: #a5b4fc;}
.ed-hero .v {font-size: 1.45rem; font-weight: 750;}
.ed-hero .v small {font-size: .85rem; font-weight: 600; margin-left: 6px;}
.ed-pill {display: inline-block; padding: 6px 18px; border-radius: 999px; font-weight: 800; font-size: 1.05rem;
  color: #fff; letter-spacing: .02em;}
.ed-bar {width: 150px; height: 8px; border-radius: 99px; background: rgba(255,255,255,.22); overflow: hidden;
  margin-top: 6px;}
.ed-bar > div {height: 100%; background: #93c5fd;}
.ed-chip {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 700;
  margin: 0 6px 4px 0; border: 1px solid transparent; white-space: nowrap;}
.ed-card {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 15px 18px; margin-bottom: 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);}
.ed-card h4 {margin: 0 0 6px; font-size: 1rem;}
.ed-card .meta {color: #64748b; font-size: .8rem; margin-bottom: 6px;}
.ed-card p {margin: 4px 0; font-size: .92rem; line-height: 1.5;}
.ed-card ul {margin: 4px 0 0 18px; padding: 0; font-size: .88rem;}
.ed-card a {color: #1d4ed8; text-decoration: none; font-weight: 600;}
.ed-accent {border-left: 4px solid #1d4ed8;}
.ed-accent.green {border-left-color: #15803d;} .ed-accent.red {border-left-color: #b91c1c;}
.ed-accent.amber {border-left-color: #b45309;} .ed-accent.slate {border-left-color: #94a3b8;}
.ed-cite {font-family: ui-monospace, Menlo, monospace; font-size: .72rem; background: #eef2ff; color: #3730a3;
  padding: 0 5px; border-radius: 5px; margin-left: 2px; white-space: nowrap;}
.ed-news {display: flex; gap: 12px; align-items: flex-start; padding: 11px 4px; border-bottom: 1px solid #edf1f7;}
.ed-news .when {min-width: 118px; color: #64748b; font-size: .78rem; padding-top: 2px;}
.ed-news .body {flex: 1;}
.ed-news .title {font-weight: 650; font-size: .95rem; color: #0f172a;}
.ed-news .title a {color: #0f172a; text-decoration: none;} .ed-news .title a:hover {color: #1d4ed8;}
.ed-news .sub {color: #64748b; font-size: .8rem; margin-top: 3px;}
.ed-muted {color: #94a3b8;}
.ed-figure {font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-top: 8px; letter-spacing: -.01em;}
.ed-metric {font-size: .86rem; color: #334155; margin-bottom: 4px;}
.ed-section {font-size: .74rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: #64748b;
  margin: 14px 0 6px;}
.ed-feature {background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px; height: 100%;}
.ed-feature .n {font-weight: 800; color: #1d4ed8; font-size: .8rem; letter-spacing: .08em;}
.ed-feature .h {font-weight: 750; margin: 4px 0;} .ed-feature .d {color: #475569; font-size: .88rem;}
</style>
"""

CHIP = {"green": ("#dcfce7", "#166534"), "red": ("#fee2e2", "#991b1b"), "amber": ("#fef3c7", "#92400e"),
        "slate": ("#f1f5f9", "#334155"), "blue": ("#dbeafe", "#1e40af"), "violet": ("#ede9fe", "#5b21b6"),
        "new": ("#dc2626", "#ffffff")}
RATING_COLOR = {"STRONG_BUY": "#166534", "BUY": GREEN, "HOLD": SLATE, "SELL": RED, "STRONG_SELL": "#7f1d1d"}
TONE = {"positive": "green", "negative": "red", "neutral": "slate", "bullish": "green", "bearish": "red"}
MATERIALITY = {"high": "red", "medium": "amber", "low": "slate"}


def style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def rich(text: str) -> str:
    """Escaped text with citations rendered as small id tags."""
    return CITE.sub(lambda m: "".join(f'<span class="ed-cite">{i.strip()}</span>' for i in re.split(r"[,;]", m.group(1))),
                    esc(text))


def link(url: str, label: str = "source") -> str:
    u = normalize_url(url or "")
    return f'<a href="{esc(u)}" target="_blank" rel="noopener noreferrer">{esc(label)} ↗</a>' if u else ""


def chip(text: str, tone: str = "slate") -> str:
    bg, fg = CHIP.get(tone, CHIP["slate"])
    return f'<span class="ed-chip" style="background:{bg};color:{fg}">{esc(text)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def signed(v: float | None, unit: str = "%") -> str:
    if v is None:
        return "n/a"
    return f"{v:+.1f}{unit}"


def hero(rep: dict, facts: dict) -> None:
    d, g, u = rep["decision"], rep.get("grounding") or {}, rep.get("usage") or {}
    company = rep.get("company") or {}
    r1m = facts.get("T.ret_1m").numeric if facts.get("T.ret_1m") else None
    color = RATING_COLOR[d["rating"]]
    chips = "".join(chip(x, "blue") for x in (company.get("sector"), company.get("industry")) if x)
    ok = g.get("passed")
    html_block(f"""
<div class="ed-hero">
  <div class="t">{esc(company.get('name', rep['ticker']))} <span style="opacity:.7">· {esc(rep['ticker'])}</span></div>
  <div class="s">{chips} data as of {esc(rep.get('as_of'))} · source {esc(rep.get('data_source'))} · run {esc(rep['run_id'])}</div>
  <div class="row">
    <div><div class="k">Desk rating</div><span class="ed-pill" style="background:{color}">{esc(RATING_LABEL[d['rating']])}</span></div>
    <div><div class="k">Conviction</div><div class="v">{d['conviction']}<small>/ 100</small></div>
      <div class="ed-bar"><div style="width:{max(0, min(100, d['conviction']))}%"></div></div></div>
    <div><div class="k">Horizon</div><div class="v">{d['horizon_months']}<small>months</small></div></div>
    <div><div class="k">Last close</div><div class="v">${rep.get('price', 0):,.2f}
      <small style="color:{'#86efac' if (r1m or 0) >= 0 else '#fca5a5'}">{signed(r1m)} 1m</small></div></div>
    <div><div class="k">Fact check</div><div class="v">{'✓ Passed' if ok else '✗ Failed'}
      <small>{g.get('citation_coverage', 0):.0%} cited</small></div></div>
    <div><div class="k">Run cost</div><div class="v">${u.get('cost_usd', 0):.3f}<small>{u.get('tokens', 0):,} tok</small></div></div>
  </div>
</div>""")


def scorecard(analysts: dict, decision: dict, titles: dict) -> go.Figure:
    """Each analyst's score next to the PM's rating, on one -1..+1 axis."""
    names, scores, conf = [], [], []
    for k, a in analysts.items():
        if a.get("failed"):
            continue
        names.append(titles.get(k, k))
        scores.append(a["score"])
        conf.append(a["confidence"])
    colors = [GREEN if s > 0.15 else RED if s < -0.15 else SLATE for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=names, orientation="h", marker_color=colors,
                           text=[f"{s:+.2f} · conf {c}" for s, c in zip(scores, conf)], textposition="outside",
                           hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    pm = RATING_SCORE[decision["rating"]] / 2
    fig.add_vline(x=pm, line=dict(color=RATING_COLOR[decision["rating"]], width=3, dash="dot"),
                  annotation_text=f"PM: {RATING_LABEL[decision['rating']]}", annotation_position="top")
    fig.add_vline(x=0, line=dict(color="#cbd5e1", width=1))
    fig.update_layout(height=60 + 42 * len(names), margin=dict(l=10, r=40, t=30, b=10), xaxis=dict(range=[-1.15, 1.15],
                      tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["Bearish", "", "Neutral", "", "Bullish"]),
                      yaxis=dict(autorange="reversed"), plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    return fig


def point_list(points: list[dict]) -> str:
    return "<ul>" + "".join(f"<li>{rich(p['claim'])}</li>" for p in points or []) + "</ul>" if points else ""


def card(title: str, body: str, accent: str = "", meta: str = "") -> None:
    cls = f"ed-card ed-accent {accent}" if accent else "ed-card"
    html_block(f'<div class="{cls}"><h4>{title}</h4>' + (f'<div class="meta">{meta}</div>' if meta else "")
               + body + "</div>")


def note_card(n: dict) -> None:
    tone = TONE.get(n.get("stance", ""), "slate")
    tgt = ""
    if n.get("price_target"):
        tgt = (f"target ${n['prior_target']:,.0f} → ${n['price_target']:,.0f}" if n.get("prior_target")
               else f"target ${n['price_target']:,.0f}")
    chips = chip(n.get("stance", "").title() or "n/a", tone) + chip(n.get("kind", ""), "violet") + \
        (chip(n["rating"], "blue") if n.get("rating") else "") + (chip(tgt, "amber") if tgt else "") + \
        (chip("ratings table: no reasoning", "slate") if n.get("aggregator") else "")
    pts = "<ul>" + "".join(f"<li>{esc(p)}</li>" for p in n.get("key_points") or []) + "</ul>" if n.get("key_points") else ""
    card(esc(n.get("source", "Research")) + (f' <span class="ed-cite">{esc(n["fact_id"])}</span>' if n.get("fact_id") else ""),
         f"<div>{chips}</div><p>{esc(n.get('thesis', ''))}</p>{pts}",
         accent={"green": "green", "red": "red"}.get(tone, "slate"),
         meta=f"{esc(n.get('date', ''))} · {esc(n.get('domain', ''))} · {link(n.get('url', ''), n.get('title') or 'read')}")


DRIVER_TONE = {"company guidance": "blue", "customer spending": "green", "market size": "violet",
               "supply and capacity": "amber", "pricing": "amber", "competition": "red", "expectations": "blue",
               "regulation": "red", "other": "slate"}


def driver_card(d: dict) -> None:
    """One quantified industry or demand driver found on the web, with its verified source."""
    kind = d.get("kind", "other")
    head = (f'<div class="ed-figure">{esc(d.get("figure", ""))}</div>'
            f'<div class="ed-metric">{esc(d.get("metric", ""))}'
            + (f' <span class="ed-muted">· {esc(d["period"])}</span>' if d.get("period") else "") + "</div>")
    body = f"<div>{chip(kind, DRIVER_TONE.get(kind, 'slate'))}</div>{head}" + \
        (f"<p>{esc(d['context'])}</p>" if d.get("context") else "")
    card(esc(d.get("source", "Source")) + (f' <span class="ed-cite">{esc(d["fact_id"])}</span>' if d.get("fact_id") else ""),
         body, accent="slate",
         meta=f"{esc(d.get('date', ''))} · {esc(d.get('domain', ''))} · {link(d.get('url', ''), d.get('title') or 'read')}")


def when_text(value: str) -> str:
    """A publish time for display: 'YYYY-MM-DD HH:MM' in this machine's local time zone, or the date alone
    when the source gave no time."""
    from datetime import datetime, timezone
    v = (value or "").strip()
    if len(v) <= 10:
        return v
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return v[:16]


def news_row(when: str, title: str, url: str, sub: str, chips_html: str = "") -> str:
    t = f'<a href="{esc(normalize_url(url))}" target="_blank" rel="noopener noreferrer">{esc(title)}</a>' \
        if normalize_url(url or "") else esc(title)
    return (f'<div class="ed-news"><div class="when">{esc(when_text(when))}</div><div class="body">'
            f'<div class="title">{t}</div><div class="sub">{chips_html}{sub}</div></div></div>')


TARGET_MOVE = re.compile(r"target \$([\d,]+) → \$([\d,]+)")


def action_tone(text: str) -> tuple[str, str]:
    """A sell-side action's label and colour: rating changes first, then the direction of a target move."""
    t = (text or "").lower()
    if "downgrade" in t:
        return "downgrade", "red"
    if "upgrade" in t:
        return "upgrade", "green"
    if "initiation" in t:
        return "initiation", "blue"
    m = TARGET_MOVE.search(text or "")
    if m:
        old, new = (float(x.replace(",", "")) for x in m.groups())
        if new < old:
            return "target cut", "amber"
        if new > old:
            return "target raised", "green"
    return "reiterated", "slate"


def page_header(title: str, subtitle: str = "") -> None:
    html_block(f'<div class="ed-hero" style="padding:16px 24px"><div class="t" style="font-size:1.5rem">{esc(title)}'
               f'</div><div class="s">{esc(subtitle)}</div></div>')


def section(label: str) -> None:
    html_block(f'<div class="ed-section">{esc(label)}</div>')
