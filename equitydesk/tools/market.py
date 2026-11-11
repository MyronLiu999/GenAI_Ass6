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
        published = c.get("pubDate") or c.get("displayTime")
        if not published and c.get("providerPublishTime"):
            published = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc).isoformat()
        out.append({"title": title, "publisher": publisher, "published": (published or "")[:10],
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
            calendar=results.get("calendar") or {}, errors=errors, source="yfinance")


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
                 ("info", "income", "cashflow", "news", "calendar")}
        for n in parts:
            if on_call:
                on_call(tool=n, ticker=ticker, ok=True, cached=True, ms=0)
        return MarketBundle(ticker=ticker, history=hist, benchmark=bench, info=parts["info"],
                            income=parts["income"], cashflow=parts["cashflow"],
                            news=parts["news"][: self.s.data.news_limit], calendar=parts["calendar"],
                            errors=errors, source="fixtures")


def make_provider(settings: Settings) -> Provider:
    return FixtureProvider(settings) if settings.data.provider == "fixtures" else YFinanceProvider(settings)


def snapshot(bundle: MarketBundle, root: Path, benchmark: str) -> Path:
    """Write a fetched bundle as a fixture directory (and the benchmark's history beside it)."""
    d = root / bundle.ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    bundle.history.to_csv(d / "history.csv")
    for name in ("info", "income", "cashflow", "news", "calendar"):
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
