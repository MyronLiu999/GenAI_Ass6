"""Market monitor: what the background service collected, scored, pushed and researched. The service runs on
its own (python -m equitydesk.monitor, start_monitor.bat, or the Start button here); this page reads its store
and refreshes itself every 30 seconds."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import streamlit as st

from equitydesk.config import load_env, load_settings
from equitydesk.memory.store import Store
from equitydesk.monitor import service as svc_mod
from equitydesk.monitor.service import MonitorService, seed_watch, status
from equitydesk.monitor.store import MonitorStore
from equitydesk.monitor.triage import CATEGORY_ZH, verification
from equitydesk.policy.guards import InputError, normalize_ticker
from equitydesk.ui import components as ui
from equitydesk.ui.common import mode

load_env()
S = load_settings()            # the monitor always works on live data, whatever the sidebar mode
store = MonitorStore(S.monitor_path)
desk_store = Store(S.store_path)
ZH = S.monitor.language == "Chinese"


def L(zh: str, en: str) -> str:
    return zh if ZH else en


INDEX_ZH = {"S&P 500": "标普500", "Nasdaq": "纳指", "PHLX Semiconductor": "费城半导体", "VIX": "VIX 恐慌指数",
            "10Y Treasury yield": "10年期美债收益率", "US Dollar Index": "美元指数", "WTI crude": "原油"}
VERIFY = {"official": (L("官方", "official"), "green"), "mainstream": (L("主流媒体", "mainstream"), "blue"),
          "single source": (L("单一来源", "single source"), "slate"), "unverified": (L("未经证实", "unverified"), "red")}


def cat(c: str | None) -> str:
    return CATEGORY_ZH.get(c or "", c or "") if ZH else (c or "")


def local_time(iso: str, fmt: str = "%H:%M") -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().strftime(fmt)
    except ValueError:
        return iso[11:16]


def service() -> MonitorService:
    return MonitorService(S, store)


if not store.watch():
    seed_watch(store, desk_store.get_profile()["watchlist"])

if mode() == "demo":
    st.info(L("市场监控始终使用实时数据（与侧边栏的离线演示模式无关）。", "The monitor always works on live data, "
              "independent of the sidebar's offline demo mode."), icon=":material/info:")


# -- header: status and controls ---------------------------------------------------------------------
@st.fragment(run_every="30s")
def header() -> None:
    stt = status(store, S.monitor.interval_s)
    state = stt["state"]
    dot, label = {"running": ("#22c55e", L("运行中", "running")), "paused": ("#f59e0b", L("已暂停", "paused")),
                  "stopped": ("#94a3b8", L("已停止", "stopped"))}.get(state, ("#ef4444", L("未运行", "not running")))
    last = ui.ago_text((store.get_with_time("last_collect")[1]), ZH)
    eng = service().engine()
    eng_label = L("Claude Code（订阅，无 API 费用）", "Claude Code (subscription, no API cost)") if eng == "claude_code" \
        else "OpenAI API"
    ui.html_block(f'<div style="display:flex;align-items:center;gap:12px;margin:4px 0 6px">'
                  f'<span style="width:14px;height:14px;border-radius:50%;background:{dot};display:inline-block"></span>'
                  f'<span style="font-size:1.6rem;font-weight:800">{L("市场监控", "Market monitor")}</span>'
                  f'<span class="ed-muted" style="font-size:1rem">{label}'
                  + (f' · {L("上次采集", "last collected")} {last}' if last else "")
                  + f' · {L("模型", "model")}: {ui.esc(eng_label)}</span></div>')
    c = st.columns([1, 1, 1, 5])
    if state in ("not running", "stopped"):
        if c[0].button(L("启动", "Start"), icon=":material/play_arrow:", type="primary"):
            store.put("control", {"state": "running"})
            svc_mod.spawn(S.data_dir / "monitor.log")
            st.toast(L("监控已在后台启动", "Monitor started in the background"))
    elif state == "running":
        if c[0].button(L("暂停", "Pause"), icon=":material/pause:"):
            store.put("control", {"state": "paused"})
            st.rerun(scope="fragment")
    elif state == "paused":
        if c[0].button(L("继续", "Resume"), icon=":material/play_arrow:", type="primary"):
            store.put("control", {"state": "running"})
            st.rerun(scope="fragment")
    if state in ("running", "paused") and c[1].button(L("停止", "Stop"), icon=":material/stop:"):
        store.put("control", {"state": "stopped"})
        st.rerun(scope="fragment")
    note = store.get("budget_note")
    err = (store.get("last_collect") or {}).get("model_error") or (store.get("model_error") or {}).get("error")
    if note:
        st.warning(note)
    if err and "not logged in" in err.lower():
        st.error(err)


header()
tabs = st.tabs([L("总览", "Overview"), L("事件流", "Event stream"), L("快照", "Snapshots"), L("自选股", "Watchlist"),
                L("来源与设置", "Sources & settings")])


# -- overview -------------------------------------------------------------------------------------------
def source_links(keys: list[str]) -> str:
    out = []
    for k in keys[:4]:
        it = store.get_item(k)
        if it:
            out.append(ui.link(it.get("url", ""), it["source"]) or ui.esc(it["source"]))
    return " · ".join(out)


@st.fragment(run_every="30s")
def overview_tab() -> None:
    n = store.counts_today()
    spend = store.spend_today()
    k = st.columns(6)
    k[0].metric(L("今日新闻", "Items today"), n["items"])
    k[1].metric(L("已分诊", "Scored"), n["scored"])
    k[2].metric(L("高分 ≥8", "Score ≥8"), n["high"])
    k[3].metric(L("已推送", "Pushed"), n["pushed"])
    k[4].metric(L("调研报告", "Reports"), f"{n['reports']}/{S.monitor.max_reports_per_day}")
    k[5].metric(L("估算花费", "Est. spend"), f"${spend:.2f}" if service().engine() == "openai" else L("订阅", "subscription"))

    watch = store.watch()
    day = store.items(hours=24)
    chips = []
    for w in watch:
        mine = [i for i in day if w["ticker"] in (i.get("tickers") or [])]
        top = max((i.get("score") or 0 for i in mine), default=0)
        color = "#f97316" if top >= 8 else "#facc15" if top >= 7 else "#cbd5e1"
        chips.append(f'<span style="display:inline-flex;gap:8px;align-items:center;padding:6px 12px;margin:3px;'
                     f'border-radius:8px;border:1px solid #e2e8f0;border-left:4px solid {color};font-family:ui-monospace,'
                     f'monospace;font-weight:700">{ui.esc(w["ticker"])}<span class="ed-muted">{len(mine)}</span></span>')
    if chips:
        ui.html_block('<div style="margin:8px 0 14px">' + "".join(chips) + "</div>")

    q, q_at = store.get_with_time("quotes")
    if q and q.get("indexes"):
        tiles = []
        for x in q["indexes"]:
            ch = x.get("change_pct")
            col = "#15803d" if (ch or 0) >= 0 else "#dc2626"
            last = f"{x['last']:,.2f}"
            tiles.append(f'<div style="padding:12px 14px;border:1px solid #edf1f7;border-radius:10px;background:#fff">'
                         f'<div class="ed-muted" style="font-size:.82rem">{ui.esc(INDEX_ZH.get(x["name"], x["name"]) if ZH else x["name"])}</div>'
                         f'<div style="font-weight:800;font-size:1.15rem">{last} '
                         f'<span style="color:{col};font-size:.9rem">{ui.signed(ch, "%") if ch is not None else ""}</span></div>'
                         f'{ui.sparkline(x.get("spark") or [], up=(ch or 0) >= 0)}</div>')
        quote_age = ui.ago_text(max((x.get("time") or "") for x in q["indexes"]), ZH)
        ui.html_block(f'<div class="ed-card"><h4>{L("行情", "Markets")} <span class="ed-muted" style="font-weight:400;'
                      f'font-size:.85rem">{L("报价时间", "quoted")} {quote_age} · {L("可能延迟", "may be delayed")}</span></h4>'
                      '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px">'
                      + "".join(tiles) + "</div></div>")

    ov, ov_at = store.get_with_time("overview")
    c1, c2 = st.columns([5, 1])
    with c2:
        if st.button(L("重新生成", "Regenerate"), icon=":material/refresh:", key="regen"):
            s = service()
            llm = s._llm()
            if llm is None:
                st.warning(L("今日花费已达上限", "Daily cost ceiling reached"))
            else:
                with st.spinner(L("生成中…", "Writing…")):
                    try:
                        s.overview_due(llm, store.watch(), force=True)
                    finally:
                        store.add_spend(llm.meter.cost_usd)
                st.rerun(scope="fragment")
    if ov:
        def block(title, points):
            if not points:
                return ""
            li = "".join(f'<li>{ui.esc(p["text"])} <span class="ed-muted" style="font-size:.78rem">'
                         f'{source_links(p.get("keys") or [])}</span></li>' for p in points)
            return f'<div class="ed-section">{title}</div><ul>{li}</ul>'
        body = block(L("宏观", "Macro"), ov.get("macro"))
        for t in ov.get("tickers") or []:
            body += block(t["ticker"], t["points"])
        body += block(L("接下来关注", "Watch next"), ov.get("watch_next"))
        with c1:
            ui.html_block(f'<div class="ed-card"><h4>{L("AI 总览", "AI overview")} <span class="ed-muted" '
                          f'style="font-weight:400;font-size:.85rem">{ui.ago_text(ov_at, ZH)}'
                          f'{L("生成", " generated")}</span></h4>{body}</div>')
    else:
        with c1:
            st.info(L("总览会在第一批新闻打分后自动生成。", "The overview is written once the first items are scored."))

    odds = store.get("predictions") or []
    if odds:
        rows = []
        for e in odds:
            outs = " ".join(
                f'<span style="margin-right:14px">{ui.esc(o["label"])} <b>{o["prob"]:.0f}%</b>'
                + (f' <span style="color:{"#15803d" if o["change"] > 0 else "#dc2626"}">({o["change"]:+.0f})</span>'
                   if o.get("change") not in (None, 0) and abs(o["change"]) >= 0.5 else "") + "</span>"
                for o in e["outcomes"])
            tick = f'<b style="font-family:ui-monospace,monospace;margin-right:8px">{ui.esc(e["ticker"])}</b>' \
                if e.get("ticker") and e["venue"] == "Polymarket" else ""
            rows.append(f'<div style="padding:10px 2px;border-bottom:1px solid #edf1f7">'
                        f'<div style="display:flex;justify-content:space-between;gap:10px"><div>{tick}'
                        f'<a href="{ui.esc(e["url"])}" target="_blank" rel="noopener noreferrer" style="font-weight:700;'
                        f'color:#0f172a;text-decoration:none">{ui.esc(e["title"])}</a></div>'
                        f'<div class="ed-muted" style="font-size:.8rem;white-space:nowrap">{e["venue"]} · '
                        f'{L("日成交", "24h vol")} ${e["volume24h"]:,.0f}</div></div><div style="margin-top:4px">{outs}</div></div>')
        ui.html_block(f'<div class="ed-card"><h4>{L("预测市场", "Prediction markets")} <span class="ed-muted" '
                      f'style="font-weight:400;font-size:.85rem">Kalshi · Polymarket '
                      f'{L("下注价格隐含的概率，不是事实 · 括号内为一天变动", "odds implied by bets, not facts · (1-day change)")}'
                      f'</span></h4>{"".join(rows)}</div>')

    reps = store.reports(10)
    if reps:
        ui.section(L("调研报告", "Event reports"))
        for r in reps:
            with st.expander(f"{local_time(r['created_at'], '%m-%d %H:%M')} · {r['title']} · "
                             f"{len(r['sources'])} {L('个来源', 'sources')}"):
                ui.html_block(report_html(r))


def report_html(r: dict) -> str:
    by = {s["id"]: s for s in r["sources"]}
    body = ui.esc(r["body"]).replace("\n", "<br>")
    body = re.sub(r"\[(S\d+)\]", lambda m: f'<sup>{ui.link(by[m.group(1)]["url"], m.group(1))}</sup>'
                  if m.group(1) in by else "", body)
    src = "".join(f'<li>[{s["id"]}] {ui.link(s["url"], s["title"] or s["domain"])} <span class="ed-muted">'
                  f'{ui.esc(s["domain"])}</span></li>' for s in r["sources"])
    return (f'<div style="font-size:.92rem;line-height:1.6">{body}</div>'
            + (f'<div class="ed-section">{L("来源", "Sources")}</div><ol style="list-style:none;padding-left:0">{src}</ol>'
               if src else ""))


# -- event stream ---------------------------------------------------------------------------------------
def item_row(it: dict, holdings: set[str]) -> str:
    v = verification(it)
    vlabel, vtone = VERIFY.get(v, (v, "slate"))
    chips = "".join(ui.chip(t, "blue") for t in it.get("tickers") or [])
    chips += ui.chip(vlabel, vtone) if v != "single source" else ""
    if (it.get("n_sources") or 1) > 1:
        chips += ui.chip(f'{it["n_sources"]} {L("家来源", "sources")}', "violet")
    if it.get("flagged"):
        chips += ui.chip(L("含可疑指令文本", "injection text flagged"), "red")
    title = ui.esc(it.get("title_t") or it["title"])
    if it.get("url"):
        title = f'<a href="{ui.esc(it["url"])}" target="_blank" rel="noopener noreferrer" style="color:#0f172a;' \
                f'text-decoration:none">{title}</a>'
    orig = f'<div class="ed-muted" style="font-size:.78rem">{ui.esc(it["title"])}</div>' \
        if it.get("title_t") and it["title_t"] != it["title"] else ""
    rep = ""
    if it.get("report_id"):
        r = store.get_report(it["report_id"])
        if r:
            rep = f'<details style="margin-top:6px"><summary style="cursor:pointer;color:#1d4ed8">' \
                  f'{L("调研报告", "Report")}</summary>{report_html(r)}</details>'
    return (f'<div style="display:flex;gap:14px;padding:12px 4px;border-bottom:1px solid #edf1f7">'
            f'<div>{ui.score_badge(it.get("score"))}</div><div style="flex:1">'
            f'<div style="font-weight:700;font-size:1rem">{title}</div>{orig}'
            f'<div class="ed-muted" style="font-size:.82rem;margin-top:3px">{local_time(it["published"])} · '
            f'{ui.esc(it["source"])} · {ui.esc(cat(it.get("category")))} {chips}</div>{rep}</div></div>')


def day_label(iso: str) -> str:
    d = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().date()
    today = datetime.now().date()
    return L("今天", "Today") if d == today else L("昨天", "Yesterday") if d == today - timedelta(days=1) \
        else d.isoformat()


@st.fragment(run_every="30s")
def stream_tab() -> None:
    c = st.columns([2.2, 2, 1.2, 1.2])
    lvl = c[0].segmented_control(L("分数", "Score"), ["≥8", "≥7", "≥5", L("全部", "all")], default="≥5", key="lvl")
    win = c[1].segmented_control(L("时间", "Window"), [L("6 小时", "6h"), L("24 小时", "24h"), L("3 天", "3d")],
                                 default=L("24 小时", "24h"), key="win")
    hold = c[2].checkbox(L("只看持仓", "Holdings only"), key="hold")
    with_rep = c[3].checkbox(L("只看有报告", "With report"), key="withrep")
    q = st.text_input(L("搜索标题、代码…", "Search title, ticker…"), key="q", label_visibility="collapsed",
                      placeholder=L("搜索标题、代码…", "Search title, ticker…"))
    hours = {L("6 小时", "6h"): 6, L("24 小时", "24h"): 24, L("3 天", "3d"): 72}.get(win or "", 24)
    min_score = {"≥8": 8, "≥7": 7, "≥5": 5}.get(lvl or "", None)
    holdings = {w["ticker"] for w in store.watch() if w["holding"]}
    rows = [i for i in store.items(hours=hours, min_score=min_score) if i.get("scored_at") or min_score is None]
    if hold:
        rows = [i for i in rows if set(i.get("tickers") or []) & holdings]
    if with_rep:
        rows = [i for i in rows if i.get("report_id")]
    if q:
        ql = q.lower()
        rows = [i for i in rows if ql in (i["title"] + " " + (i.get("title_t") or "")).lower()
                or q.upper() in (i.get("tickers") or [])]
    if not rows:
        st.info(L("没有符合条件的消息。", "Nothing matches."))
        return
    out, last = [], None
    for it in rows[:400]:
        d = day_label(it["published"])
        if d != last:
            out.append(f'<div class="ed-muted" style="padding:10px 4px 2px;font-size:.85rem">{d}</div>')
            last = d
        out.append(item_row(it, holdings))
    ui.html_block('<div class="ed-card">' + "".join(out) + "</div>")
    if len(rows) > 400:
        st.caption(L(f"仅显示最新 400 条（共 {len(rows)} 条）。", f"Showing the latest 400 of {len(rows)}."))


# -- snapshots --------------------------------------------------------------------------------------------
@st.fragment(run_every="60s")
def snapshot_tab() -> None:
    watch = store.watch()
    day = store.items(hours=24)
    quotes = (store.get("quotes") or {}).get("stocks") or {}
    reddit = store.get("reddit") or {}
    tw = store.get("stocktwits") or {}
    cols = st.columns(2)
    for n, w in enumerate(watch):
        t = w["ticker"]
        mine = sorted([i for i in day if t in (i.get("tickers") or [])],
                      key=lambda i: (-(i.get("score") or 0), i["published"]))
        top = max((i.get("score") or 0 for i in mine), default=0)
        qt = quotes.get(t) or {}
        ch = qt.get("change_pct")
        price = (f'<span style="font-weight:800">{qt["last"]:,.2f}</span> <span style="color:'
                 f'{"#15803d" if (ch or 0) >= 0 else "#dc2626"};font-weight:700">{ui.signed(ch, "%")}</span>') if qt else ""
        r = reddit.get(t)
        social = ""
        if r:
            rc = f' <span style="color:{"#15803d" if (r["change_pct"] or 0) >= 0 else "#dc2626"}">' \
                 f'{"↑" if (r["change_pct"] or 0) >= 0 else "↓"}{abs(r["change_pct"] or 0):.0f}%</span>' \
                if r.get("change_pct") is not None else ""
            social += f'Reddit <b>{r["mentions"]}</b> {L("次提及", "mentions")} · {L("第", "#")}{r["rank"]}' \
                      f'{L(" 名", "")}{rc}'
        s = tw.get(t)
        if s and (s["bull"] or s["bear"]):
            social += (f' &nbsp; StockTwits {L("多", "bull")} <b>{s["bull"]}</b> / {L("空", "bear")} <b>{s["bear"]}</b> '
                       + ui.bull_bear_bar(s["bull"], s["bear"]))
        color = "#f97316" if top >= 8 else "#facc15" if top >= 7 else "#cbd5e1"
        items_html = "".join(
            f'<div style="display:flex;gap:10px;padding:8px 0;border-top:1px solid #f1f5f9">{ui.score_badge(i.get("score"))}'
            f'<div><div style="font-weight:650">{ui.esc(i.get("title_t") or i["title"])}</div><div class="ed-muted" '
            f'style="font-size:.78rem">{ui.ago_text(i["published"], ZH)} · {ui.esc(i["source"])}</div></div></div>'
            for i in mine[:4])
        with cols[n % 2]:
            ui.html_block(
                f'<div class="ed-card" style="border-left:4px solid {color}"><div style="display:flex;justify-content:'
                f'space-between"><div><b style="font-family:ui-monospace,monospace;font-size:1.1rem">{ui.esc(t)}</b> '
                f'<span class="ed-muted">{len(mine)} {L("条", "items")} · {L("最高", "top")} {top} {L("分", "")}</span>'
                + (f' {ui.chip(L("持仓", "holding"), "green")}' if w["holding"] else "")
                + f'</div><div>{price}</div></div><div class="ed-muted" style="font-size:.85rem;margin:4px 0 6px">'
                  f'{social}</div>{items_html or "<div class=ed-muted>" + L("24 小时内无相关消息", "No items in 24h") + "</div>"}</div>')
            if len(mine) > 4:
                with st.expander(L(f"查看全部 {len(mine)} 条", f"All {len(mine)} items")):
                    ui.html_block("".join(item_row(i, set()) for i in mine))


# -- watchlist ----------------------------------------------------------------------------------------------
def sync_profile() -> None:
    """The desk's watchlist (Watchlist page, news in the Research desk) follows the monitor's."""
    desk_store.set_profile({"watchlist": [w["ticker"] for w in store.watch()]})


def watch_tab() -> None:
    with st.form("add_watch", clear_on_submit=True, border=True):
        st.markdown(f'**{L("添加股票", "Add a stock")}** <span class="ed-muted">'
                    f'{L("输入美股代码，例如 NVDA、AMD、BRK-B", "a US ticker, e.g. NVDA, AMD, BRK-B")}</span>',
                    unsafe_allow_html=True)
        a, b = st.columns([5, 1])
        raw = a.text_input(L("股票代码", "Ticker"), label_visibility="collapsed", placeholder=L("股票代码", "Ticker"))
        if b.form_submit_button(L("添加", "Add"), type="primary", width="stretch") and raw:
            try:
                t = normalize_ticker(raw)
                if t in {w["ticker"] for w in store.watch()}:
                    st.info(L(f"{t} 已在自选股中", f"{t} is already on the list"))
                else:
                    store.upsert_watch(t, aliases=svc_mod.default_aliases(t))
                    sync_profile()
                    st.toast(L(f"已添加 {t}", f"Added {t}"))
            except InputError as e:
                st.error(str(e))
    watch = store.watch()
    st.markdown(f'**{L("我的自选", "My watchlist")}** <span class="ed-muted">{len(watch)} {L("只", "stocks")}</span>',
                unsafe_allow_html=True)
    st.caption(L("别名用逗号分隔（公司名、中文名等）；新闻中出现任一别名即归入该股票。勾选“持仓”后，相关消息推送门槛更低，"
                 "并可在事件流中只看持仓。", "Aliases are comma-separated (company name, Chinese name...): a headline "
                 "naming any of them is tagged with the stock. Holdings get a lower push threshold and their own filter."))
    for w in watch:
        t = w["ticker"]
        c = st.columns([1.1, 4, 1.4, 1, 1])
        c[0].markdown(f'<div style="padding-top:8px;font-family:ui-monospace,monospace;font-weight:800">{ui.esc(t)}</div>',
                      unsafe_allow_html=True)
        al = c[1].text_input("aliases", ", ".join(w["aliases"]), key=f"al_{t}", label_visibility="collapsed")
        mt = c[2].checkbox(L("按代码匹配", "Match ticker"), w["match_ticker"], key=f"mt_{t}")
        hd = c[3].checkbox(L("持仓", "Holding"), w["holding"], key=f"hd_{t}")
        new_al = [x.strip() for x in re.split(r"[,，]", al) if x.strip()]
        if new_al != w["aliases"] or mt != w["match_ticker"] or hd != w["holding"]:
            store.upsert_watch(t, aliases=new_al, match_ticker=mt, holding=hd)
        if c[4].button(L("移除", "Remove"), key=f"rm_{t}"):
            store.remove_watch(t)
            sync_profile()
            st.rerun()


# -- sources & settings ---------------------------------------------------------------------------------------
def settings_tab() -> None:
    s = service()
    st.markdown(f"**{L('模型', 'Model')}**")
    eng = st.radio(L("用什么模型给新闻打分、写总览和报告", "Which model scores items and writes the overview and reports"),
                   ["openai", "claude_code"], index=0 if s.engine() == "openai" else 1, horizontal=True,
                   format_func=lambda e: "OpenAI API (gpt-4o-mini / gpt-4o)" if e == "openai"
                   else L("Claude Code（本机已登录的 CLI，订阅内，无 API 费用）",
                          "Claude Code (the CLI logged in on this machine; covered by the subscription)"))
    if eng != s.engine():
        store.put("engine", eng)
        st.toast(L("已切换，下一轮采集生效", "Switched; applies from the next tick"))
    if eng == "claude_code":
        from equitydesk.harness.claude_code import find_cli
        exe = find_cli()
        st.caption((L("找到 Claude Code：", "Claude Code found: ") + f"`{exe}`") if exe else
                   L("没有找到 Claude Code CLI。", "Claude Code CLI not found."))
        if exe:
            st.caption(L("第一次使用前，在终端运行一次下面的命令登录（会打开浏览器）：",
                         "Before first use, log in once from a terminal (opens the browser):"))
            st.code(f'"{exe}" auth login', language="bash")

    st.markdown(f"**{L('推送', 'Push')}**")
    ncfg = s.notify_cfg()
    c = st.columns([1.3, 3, 1.4])
    desk_on = c[0].toggle(L("Windows 桌面通知", "Windows notifications"), ncfg.desktop)
    topic = c[1].text_input(L("ntfy 主题（手机推送，可选）", "ntfy topic (phone push, optional)"), ncfg.ntfy_topic,
                            help=L("在手机上安装 ntfy 应用并订阅同一个主题。主题名就是密码：用一个难猜的名字。",
                                   "Install the ntfy app on your phone and subscribe to the same topic. The topic name "
                                   "is the password: make it hard to guess."))
    if desk_on != ncfg.desktop or topic != ncfg.ntfy_topic:
        store.put("notify", {"desktop": desk_on, "ntfy_topic": topic.strip()})
    if c[2].button(L("发送测试推送", "Send a test push"), icon=":material/notifications:"):
        from equitydesk.monitor import notify
        sent = notify.push(s.notify_cfg(), {"score": 8, "tickers": [], "title": "EquityDesk",
                                             "title_t": L("测试推送：监控工作正常", "Test push: the monitor works"),
                                             "source": "EquityDesk", "published": datetime.now(timezone.utc).isoformat(),
                                             "why": ""}, "official")
        st.toast((L("已发送到：", "Sent to: ") + ", ".join(sent)) if sent else L("没有可用的推送渠道", "No channel accepted it"))
    st.caption(L(f"推送规则：≥{S.monitor.push_min_score} 分，或持仓相关且 ≥{S.monitor.push_holding_min_score} 分；"
                 f"每小时最多 {S.monitor.max_pushes_per_hour} 条；≥{S.monitor.report_min_score} 分自动联网写报告，"
                 f"每天最多 {S.monitor.max_reports_per_day} 份。",
                 f"Push: score ≥{S.monitor.push_min_score}, or ≥{S.monitor.push_holding_min_score} for a holding; at most "
                 f"{S.monitor.max_pushes_per_hour} per hour. Reports: score ≥{S.monitor.report_min_score}, at most "
                 f"{S.monitor.max_reports_per_day} a day. Change these in settings.yaml."))

    st.markdown(f"**{L('开机自启动', 'Start with Windows')}**")
    on = svc_mod.autostart_installed()
    want = st.toggle(L("登录 Windows 时在后台自动启动监控", "Start the monitor in the background when I log in"), on)
    if want != on:
        (svc_mod.install_autostart if want else svc_mod.remove_autostart)()
        st.toast(L("已设置", "Done"))

    st.markdown(f"**{L('数据来源', 'Sources')}**")
    from equitydesk.monitor.sources import SOURCES
    health = store.get("health") or {}
    tiers = {s_.key: (s_.name, s_.tier) for s_ in SOURCES}
    rows = []
    for key, h in sorted(health.items()):
        name, tier = tiers.get(key, (key.replace("yahoo_", "Yahoo ").replace("signal:", ""),
                                     "signal" if key.startswith("signal:") else "other"))
        rows.append({L("来源", "source"): name, L("类别", "tier"): tier,
                     L("上次成功", "last ok"): ui.ago_text(h.get("ok_at"), ZH),
                     L("条数", "items"): h.get("n"), L("错误", "error"): h.get("error") or ""})
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(L("官方 = 机构原文（美联储、SEC、白宫、BLS）；主流媒体 = 一手新闻机构；other = 转述/聚合（华尔街见闻 7x24、"
                 "Yahoo 个股新闻）。单一非主流来源、或内容本身是传闻的消息，会标成“未经证实”。",
                 "official = the institution itself; mainstream = a primary newsroom; other = relays and aggregators. "
                 "Items that report unconfirmed claims are marked unverified."))


with tabs[0]:
    overview_tab()
with tabs[1]:
    stream_tab()
with tabs[2]:
    snapshot_tab()
with tabs[3]:
    watch_tab()
with tabs[4]:
    settings_tab()
