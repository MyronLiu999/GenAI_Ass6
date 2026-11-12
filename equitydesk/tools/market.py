"""Market data providers: live Yahoo Finance (cached on disk) and recorded fixtures (offline, deterministic).

Every component is fetched independently; a failed component is recorded in `bundle.errors` and the
desk degrades around it instead of failing the run. Every fetch is reported to `on_call` so the
run's trace has one line per tool call.

CLI:  python -m equitydesk.tools.market snapshot AAPL MSFT   # record live data as fixtures
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from equitydesk.config import Settings, load_settings

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
ToolHook = Callable[..., None]


@dataclass
class MarketBundle:
    ticker: str
    history: pd.DataFrame
    benchmark: pd.DataFrame
    info: dict = field(default_factory=dict)
    income: dict = field(default_factory=dict)      # {item: {period: value}}
    cashflow: dict = field(default_factory=dict)
    news: list[dict] = field(default_factory=list)  # {title, publisher, published, url, summary}
    calendar: dict = field(default_factory=dict)    # {next_earnings: YYYY-MM-DD}
    street: dict = field(default_factory=dict)      # sell-side research data, see normalize_street
    earnings: dict = field(default_factory=dict)    # beat history, quarter/year estimates, quarterly results
    errors: dict[str, str] = field(default_factory=dict)
    source: str = "yfinance"

    @property
    def has_prices(self) -> bool:
        return not self.history.empty and self.history["Close"].dropna().size >= 30


class Provider(Protocol):
    name: str

    def fetch(self, ticker: str, on_call: ToolHook | None = None) -> MarketBundle: ...

    def history(self, ticker: str, period: str = "1y") -> pd.DataFrame: ...


# -- normalisation -----------------------------------------------------------------------------

def clean_history(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def statement_to_dict(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {}
    out: dict[str, dict[str, float]] = {}
    for item, row in df.iterrows():
        vals = {}
        for col, v in row.items():
            if pd.notna(v):
                key = col.date().isoformat() if hasattr(col, "date") else str(col)
                vals[key] = float(v)
        if vals:
            out[str(item)] = vals
    return out


def iso_utc(value: Any) -> str:
    """A publish time as 'YYYY-MM-DDTHH:MM:SSZ' (UTC) when the feed gives a time, else the date as given."""
    if value in (None, ""):
        return ""
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
        else:
            text = str(value).strip()
            if len(text) <= 10:
                return text
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError, OverflowError):
        return str(value)[:10]


def normalize_news(items: list[dict] | None, limit: int) -> list[dict]:
    """Yahoo's news payload has changed shape over time; accept both the nested and the flat form."""
    out = []
    for raw in items or []:
        c = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        title = (c.get("title") or "").strip()
        if not title:
            continue
        provider = c.get("provider") or {}
        publisher = provider.get("displayName") if isinstance(provider, dict) else None
        publisher = publisher or c.get("publisher") or "unknown"
        url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
               or c.get("link") or "")
        published = iso_utc(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"))
        out.append({"title": title, "publisher": publisher, "published": published,
                    "url": url, "summary": (c.get("summary") or "")[:600]})
        if len(out) >= limit:
            break
    return out


def normalize_calendar(cal: Any) -> dict:
    if not isinstance(cal, dict):
        return {}
    ed = cal.get("Earnings Date")
    if isinstance(ed, (list, tuple)) and ed:
        ed = ed[0]
    if isinstance(ed, (date, datetime)):
        return {"next_earnings": ed.isoformat()[:10]}
    return {}


ACTION_LABEL = {"up": "upgrade", "down": "downgrade", "init": "initiation", "main": "maintains",
                "reit": "reiterates"}


def _frame_to_nested(df: pd.DataFrame | None) -> dict:
    """{row: {column: value}} with numeric values only (JSON-safe)."""
    if df is None or getattr(df, "empty", True):
        return {}
    out: dict[str, dict[str, float]] = {}
    for idx, row in df.iterrows():
        vals = {str(c): float(v) for c, v in row.items() if isinstance(v, (int, float)) and pd.notna(v)}
        if vals:
            out[str(idx)] = vals
    return out


def normalize_street(tk: Any) -> dict:
    """Sell-side research from Yahoo: dated rating and price-target actions by firm, the target range,
    the EPS estimate trend and revision counts, growth estimates, and the recommendation mix over time.
    Each part is read on its own; a missing part is simply absent."""
    out: dict[str, Any] = {}
    try:
        ud = tk.upgrades_downgrades
        if ud is not None and not ud.empty:
            ud = ud.sort_index(ascending=False).head(80)
            out["actions"] = [{
                "date": str(ts)[:10], "firm": str(r.get("Firm") or ""), "to": str(r.get("ToGrade") or ""),
                "from": str(r.get("FromGrade") or ""), "action": str(r.get("Action") or ""),
                "pt_action": str(r.get("priceTargetAction") or ""),
                "pt": float(r["currentPriceTarget"]) if pd.notna(r.get("currentPriceTarget")) and r.get("currentPriceTarget") else None,
                "prior_pt": float(r["priorPriceTarget"]) if pd.notna(r.get("priorPriceTarget")) and r.get("priorPriceTarget") else None,
            } for ts, r in ud.iterrows()]
    except Exception:
        pass
    try:
        pt = tk.analyst_price_targets or {}
        out["price_targets"] = {k: float(v) for k, v in pt.items() if isinstance(v, (int, float)) and pd.notna(v)}
    except Exception:
        pass
    for key, attr in (("eps_trend", "eps_trend"), ("eps_revisions", "eps_revisions"), ("growth", "growth_estimates"),
                      ("earnings_estimate", "earnings_estimate")):
        try:
            nested = _frame_to_nested(getattr(tk, attr))
            if nested:
                out[key] = nested
        except Exception:
            pass
    try:
        rec = tk.recommendations
        if rec is not None and not rec.empty:
            out["recommendations"] = [{k: (str(v) if k == "period" else int(v)) for k, v in r.items()}
                                      for r in rec.to_dict("records")]
    except Exception:
        pass
    return out


QUARTER_ROWS = {   # yfinance row -> our key; first match wins per key
    "revenue": ("Total Revenue", "Operating Revenue"), "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income", "Total Operating Income As Reported"), "net_income": ("Net Income",),
    "receivables": ("Accounts Receivable", "Receivables"), "inventory": ("Inventory",),
    "fcf": ("Free Cash Flow",), "capex": ("Capital Expenditure",),
}


def normalize_earnings(tk: Any) -> dict:
    """The earnings set-up: the EPS beat/miss record, consensus for the current and next quarter and year
    (average, low, high, number of analysts), and the last six quarters of reported results (revenue, margins,
    receivables, inventory, free cash flow, capex). Each part is read on its own; a missing part is absent."""
    out: dict[str, Any] = {}
    try:
        ed = tk.get_earnings_dates(limit=12)
        if ed is not None and not ed.empty:
            out["beats"] = [{"date": str(ts)[:10], "eps_est": float(r["EPS Estimate"]),
                             "eps_actual": float(r["Reported EPS"]),
                             "surprise_pct": float(r["Surprise(%)"]) if pd.notna(r.get("Surprise(%)")) else None}
                            for ts, r in ed.sort_index(ascending=False).iterrows()
                            if pd.notna(r.get("Reported EPS")) and pd.notna(r.get("EPS Estimate"))]
    except Exception:
        pass
    for key, attr in (("eps_est", "earnings_estimate"), ("rev_est", "revenue_estimate")):
        try:
            nested = _frame_to_nested(getattr(tk, attr))
            if nested:
                out[key] = nested
        except Exception:
            pass
    periods: dict[str, dict[str, float]] = {}
    for attr in ("quarterly_income_stmt", "quarterly_balance_sheet", "quarterly_cashflow"):
        try:
            df = getattr(tk, attr)
        except Exception:
            continue
        if df is None or getattr(df, "empty", True):
            continue
        for key, names in QUARTER_ROWS.items():
            name = next((n for n in names if n in df.index), None)
            if name is None:
                continue
            for col, v in df.loc[name].items():
                if pd.notna(v):
                    periods.setdefault(str(col)[:10], {}).setdefault(key, float(v))
    if periods:
        out["quarters"] = [{"period": p, **vals} for p, vals in sorted(periods.items(), reverse=True)
                           if "revenue" in vals][:6]
    return out


def _jsonable_info(info: dict) -> dict:
    return {k: v for k, v in (info or {}).items() if isinstance(v, (str, int, float, bool)) or v is None}


# -- disk cache --------------------------------------------------------------------------------

class DiskCache:
    """A small TTL cache of pickled values under data/cache. Only ever reads files it wrote itself."""

    def __init__(self, root: Path, ttl_minutes: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_minutes * 60

    def _path(self, key: str) -> Path:
        return self.root / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".pkl")

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists() or time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def put(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(value, f)
        tmp.replace(self._path(key))


# -- providers ---------------------------------------------------------------------------------

class YFinanceProvider:
    name = "yfinance"

    def __init__(self, settings: Settings):
        self.s = settings
        self.cache = DiskCache(settings.cache_dir, settings.data.cache_ttl_minutes)

    def _cached(self, key: str, fn: Callable[[], Any], on_call: ToolHook | None, tool: str, ticker: str) -> Any:
        hit = self.cache.get(key)
        t0 = time.perf_counter()
        if hit is not None:
            if on_call:
                on_call(tool=tool, ticker=ticker, ok=True, cached=True, ms=0)
            return hit
        try:
            value = fn()
        except Exception as e:
            if on_call:
                on_call(tool=tool, ticker=ticker, ok=False, cached=False,
                        ms=round((time.perf_counter() - t0) * 1000), error=f"{type(e).__name__}: {e}")
            raise
        if on_call:
            on_call(tool=tool, ticker=ticker, ok=True, cached=False, ms=round((time.perf_counter() - t0) * 1000))
        self.cache.put(key, value)
        return value

    def history(self, ticker: str, period: str = "1y", on_call: ToolHook | None = None) -> pd.DataFrame:
        import yfinance as yf
        day = date.today().isoformat()
        return self._cached(f"hist:{ticker}:{period}:{day}",
                            lambda: clean_history(yf.Ticker(ticker).history(period=period, auto_adjust=True)),
                            on_call, "price_history", ticker)

    def news(self, ticker: str, limit: int = 20) -> list[dict]:
        import yfinance as yf
        return self._cached(f"news:{ticker}:{date.today().isoformat()}:{time.strftime('%H')}:{limit}",
                            lambda: normalize_news(yf.Ticker(ticker).news, limit), None, "news", ticker)

    def street(self, ticker: str) -> dict:
        import yfinance as yf
        return self._cached(f"street:{ticker}:{date.today().isoformat()}:{time.strftime('%H')}",
                            lambda: normalize_street(yf.Ticker(ticker)), None, "sell_side_research", ticker)

    def info(self, ticker: str) -> dict:
        import yfinance as yf
        return self._cached(f"info:{ticker}:{date.today().isoformat()}", lambda: _jsonable_info(yf.Ticker(ticker).info),
                            None, "company_info", ticker)

    def fetch(self, ticker: str, on_call: ToolHook | None = None) -> MarketBundle:
        import yfinance as yf
        day = date.today().isoformat()
        tk = yf.Ticker(ticker)
        limit = self.s.data.news_limit
        jobs: dict[str, Callable[[], Any]] = {
            "history": lambda: self.history(ticker, self.s.data.history_period, on_call),
            "benchmark": lambda: self.history(self.s.data.benchmark, self.s.data.history_period, on_call),
            "info": lambda: self._cached(f"info:{ticker}:{day}", lambda: _jsonable_info(tk.info), on_call,
                                         "company_info", ticker),
            "income": lambda: self._cached(f"inc:{ticker}:{day}", lambda: statement_to_dict(tk.income_stmt),
                                           on_call, "income_statement", ticker),
            "cashflow": lambda: self._cached(f"cf:{ticker}:{day}", lambda: statement_to_dict(tk.cashflow),
                                             on_call, "cashflow_statement", ticker),
            "news": lambda: self._cached(f"news:{ticker}:{day}:{time.strftime('%H')}",
                                         lambda: normalize_news(tk.news, limit), on_call, "news", ticker),
            "calendar": lambda: self._cached(f"cal:{ticker}:{day}", lambda: normalize_calendar(tk.calendar),
                                             on_call, "earnings_calendar", ticker),
            "street": lambda: self._cached(f"street:{ticker}:{day}:{time.strftime('%H')}", lambda: normalize_street(tk),
                                           on_call, "sell_side_research", ticker),
            "earnings": lambda: self._cached(f"earn:{ticker}:{day}", lambda: normalize_earnings(tk), on_call,
                                             "earnings_setup", ticker),
        }
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {name: pool.submit(fn) for name, fn in jobs.items()}
            for name, fut in futures.items():
                try:
                    results[name] = fut.result()
                except Exception as e:  # each component degrades on its own
                    errors[name] = f"{type(e).__name__}: {e}"[:300]
        return MarketBundle(
            ticker=ticker,
            history=results.get("history", clean_history(None)),
            benchmark=results.get("benchmark", clean_history(None)),
            info=results.get("info") or {}, income=results.get("income") or {},
            cashflow=results.get("cashflow") or {}, news=results.get("news") or [],
            calendar=results.get("calendar") or {}, street=results.get("street") or {},
            earnings=results.get("earnings") or {}, errors=errors, source="yfinance")


class FixtureProvider:
    """Recorded data under fixtures/<TICKER>/. Deterministic and offline: tests and the eval use it."""

    name = "fixtures"

    def __init__(self, settings: Settings, root: Path | None = None):
        self.s = settings
        self.root = Path(root or settings.fixtures_dir)

    def _dir(self, ticker: str) -> Path:
        return self.root / ticker.upper()

    def available(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "history.csv").exists()) if self.root.exists() else []

    def history(self, ticker: str, period: str = "1y", on_call: ToolHook | None = None) -> pd.DataFrame:
        p = self._dir(ticker) / "history.csv"
        if not p.exists():
            if on_call:
                on_call(tool="price_history", ticker=ticker, ok=False, cached=True, ms=0, error="no fixture")
            return clean_history(None)
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if on_call:
            on_call(tool="price_history", ticker=ticker, ok=True, cached=True, ms=0)
        return clean_history(df)

    def news(self, ticker: str, limit: int = 20) -> list[dict]:
        return self._json(ticker, "news", [])[:limit]

    def street(self, ticker: str) -> dict:
        return self._json(ticker, "street", {})

    def info(self, ticker: str) -> dict:
        return self._json(ticker, "info", {})

    def _json(self, ticker: str, name: str, default: Any) -> Any:
        p = self._dir(ticker) / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def fetch(self, ticker: str, on_call: ToolHook | None = None) -> MarketBundle:
        errors = {}
        hist = self.history(ticker, on_call=on_call)
        if hist.empty:
            errors["history"] = f"no fixture for {ticker}"
        bench = self.history(self.s.data.benchmark, on_call=on_call)
        parts = {n: self._json(ticker, n, {} if n != "news" else []) for n in
                 ("info", "income", "cashflow", "news", "calendar", "street", "earnings")}
        for n in parts:
            if on_call:
                on_call(tool=n, ticker=ticker, ok=True, cached=True, ms=0)
        return MarketBundle(ticker=ticker, history=hist, benchmark=bench, info=parts["info"],
                            income=parts["income"], cashflow=parts["cashflow"],
                            news=parts["news"][: self.s.data.news_limit], calendar=parts["calendar"],
                            street=parts["street"], earnings=parts["earnings"], errors=errors, source="fixtures")


def make_provider(settings: Settings) -> Provider:
    return FixtureProvider(settings) if settings.data.provider == "fixtures" else YFinanceProvider(settings)


def snapshot(bundle: MarketBundle, root: Path, benchmark: str) -> Path:
    """Write a fetched bundle as a fixture directory (and the benchmark's history beside it)."""
    d = root / bundle.ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    bundle.history.to_csv(d / "history.csv")
    for name in ("info", "income", "cashflow", "news", "calendar", "street", "earnings"):
        (d / f"{name}.json").write_text(json.dumps(getattr(bundle, name), indent=1, default=str), encoding="utf-8")
    bd = root / benchmark.upper()
    bd.mkdir(parents=True, exist_ok=True)
    if not bundle.benchmark.empty:
        bundle.benchmark.to_csv(bd / "history.csv")
    return d


def _main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "snapshot":
        print("usage: python -m equitydesk.tools.market snapshot TICKER [TICKER ...]")
        return 2
    s = load_settings()
    prov = YFinanceProvider(s)
    for t in argv[1:]:
        b = prov.fetch(t.upper())
        path = snapshot(b, s.fixtures_dir, s.data.benchmark)
        print(f"{t.upper()}: {len(b.history)} bars, {len(b.news)} headlines -> {path}  errors={b.errors or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
