"""Market signals around the news: index quotes with an intraday sparkline, watchlist prices, Reddit mentions
(ApeWisdom), StockTwits bull/bear, and prediction-market odds (Polymarket, Kalshi). Every fetch is isolated:
a failing source leaves its last snapshot in place. Odds are prices, not facts, and are labelled as such."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from equitydesk.monitor.sources import TIMEOUT, UA

INDEXES = (("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^SOX", "PHLX Semiconductor"), ("^VIX", "VIX"),
           ("^TNX", "10Y Treasury yield"), ("DX-Y.NYB", "US Dollar Index"), ("CL=F", "WTI crude"))


def _get_json(url: str) -> Any:
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# -- quotes ----------------------------------------------------------------------------------------

def quote_from_series(close: pd.Series) -> dict | None:
    """Last price, change against the previous session's last price, and today's path for a sparkline."""
    s = close.dropna()
    if s.empty:
        return None
    days = pd.Index(s.index.date)
    last_day = days[-1]
    today = s[days == last_day]
    before = s[days < last_day]
    prev = float(before.iloc[-1]) if not before.empty else None
    last = float(s.iloc[-1])
    return {"last": last, "prev": prev, "change_pct": (last / prev - 1) * 100 if prev else None,
            "time": pd.Timestamp(s.index[-1]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            if getattr(s.index, "tz", None) is not None else str(s.index[-1]),
            "spark": [round(float(x), 4) for x in today.tolist()][-120:]}


def _closes(symbols: list[str], period: str, interval: str) -> dict[str, pd.Series]:
    import yfinance as yf
    df = yf.download(symbols, period=period, interval=interval, group_by="ticker", progress=False,
                     auto_adjust=False, threads=True)
    out = {}
    for sym in symbols:
        try:
            col = df[sym]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
            out[sym] = col
        except KeyError:
            continue
    return out


def quotes(watchlist: list[str]) -> dict:
    idx = _closes([s for s, _ in INDEXES], "5d", "15m")
    stocks = _closes(watchlist, "5d", "15m") if watchlist else {}
    return {"indexes": [{"symbol": s, "name": n, **q} for s, n in INDEXES if (q := quote_from_series(idx.get(s, pd.Series(dtype=float))))],
            "stocks": {t: q for t in watchlist if (q := quote_from_series(stocks.get(t, pd.Series(dtype=float))))}}


# -- retail sentiment --------------------------------------------------------------------------------

def reddit_mentions(get=_get_json, pages: int = 3) -> dict[str, dict]:
    """ApeWisdom: mentions across stock subreddits in the last 24 hours, rank, and the change on a day ago."""
    out = {}
    for page in range(1, pages + 1):
        data = get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}")
        for r in data.get("results") or []:
            m, m0 = int(r.get("mentions") or 0), int(r.get("mentions_24h_ago") or 0)
            out[r["ticker"]] = {"mentions": m, "rank": int(r.get("rank") or 0),
                                "rank_24h_ago": int(r.get("rank_24h_ago") or 0) or None,
                                "change_pct": (m / m0 - 1) * 100 if m0 else None}
        if page >= int(data.get("pages") or 1):
            break
    return out


def stocktwits(ticker: str, get=_get_json) -> dict:
    """Bull/bear tags among the latest StockTwits messages (users tag their own posts; most carry no tag)."""
    data = get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json")
    tags = [((m.get("entities") or {}).get("sentiment") or {}).get("basic") for m in data.get("messages") or []]
    return {"bull": tags.count("Bullish"), "bear": tags.count("Bearish"), "messages": len(tags)}


# -- prediction markets ------------------------------------------------------------------------------

def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def polymarket_events(data: dict, must_contain: str = "", min_volume: float = 5000, top: int = 3) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    out = []
    for ev in (data or {}).get("events") or []:
        if ev.get("closed") or not ev.get("active", True) or (ev.get("endDate") or "9") < now:
            continue
        if must_contain and must_contain.lower() not in (ev.get("title") or "").lower():
            continue
        vol = _f(ev.get("volume24hr")) or 0.0
        if vol < min_volume:
            continue
        outcomes = []
        for m in ev.get("markets") or []:
            # placeholder outcomes ("Candidate S") sit at 50% without a single trade: not a price
            if m.get("closed") or not (_f(m.get("volumeNum")) or _f(m.get("volume")) or 0) > 0:
                continue
            try:
                p = float(json.loads(m.get("outcomePrices") or "[]")[0])
            except (ValueError, IndexError, TypeError):
                continue
            label = m.get("groupItemTitle") or "Yes"
            ch = _f(m.get("oneDayPriceChange"))
            outcomes.append({"label": label, "prob": p * 100, "change": ch * 100 if ch is not None else None})
        outcomes.sort(key=lambda o: -o["prob"])
        if outcomes:
            out.append({"venue": "Polymarket", "title": ev.get("title", ""), "volume24h": vol,
                        "url": f"https://polymarket.com/event/{ev.get('slug', '')}", "outcomes": outcomes[:4],
                        "ends": (ev.get("endDate") or "")[:10]})
    out.sort(key=lambda e: -e["volume24h"])
    return out[:top]


def kalshi_events(data: dict, series: str, top: int = 2) -> list[dict]:
    out = []
    for ev in (data or {}).get("events") or []:
        outcomes, vol = [], 0.0
        for m in ev.get("markets") or []:
            last, prev = _f(m.get("last_price_dollars")), _f(m.get("previous_price_dollars"))
            vol += _f(m.get("volume_24h_fp")) or 0.0
            if last is None:
                continue
            outcomes.append({"label": m.get("yes_sub_title") or m.get("subtitle") or m.get("ticker", ""),
                             "prob": last * 100, "change": (last - prev) * 100 if prev is not None else None})
        outcomes.sort(key=lambda o: -o["prob"])
        if outcomes and vol > 0:
            title = ev.get("title", "") + (f" ({ev['sub_title']})" if ev.get("sub_title") else "")
            out.append({"venue": "Kalshi", "title": title, "volume24h": vol, "outcomes": outcomes[:4],
                        "url": f"https://kalshi.com/markets/{series.lower()}", "ends": ""})
    return out[:top]


def predictions(watch: list[dict], queries: list[str], series: list[str], get=_get_json) -> list[dict]:
    """Odds for the watchlist companies (searched by name) and for the configured macro questions."""
    out, seen = [], set()

    def add(events, ticker=None):
        for e in events:
            if e["title"] not in seen:
                seen.add(e["title"])
                out.append({**e, "ticker": ticker})

    for w in watch:
        name = next((a for a in w["aliases"] if a.isascii() and len(a) > 2), w["ticker"])
        try:
            data = get(f"https://gamma-api.polymarket.com/public-search?q={requests.utils.quote(name)}&limit_per_type=6")
            add(polymarket_events(data, must_contain=name.split()[0], top=2), w["ticker"])
        except Exception:
            continue
    for q in queries:
        try:
            add(polymarket_events(get(f"https://gamma-api.polymarket.com/public-search?q={requests.utils.quote(q)}"
                                      "&limit_per_type=6"), top=2))
        except Exception:
            continue
    for s in series:
        try:
            add(kalshi_events(get("https://api.elections.kalshi.com/trade-api/v2/events?status=open"
                                  f"&with_nested_markets=true&limit=3&series_ticker={s}"), s))
        except Exception:
            continue
    return out
