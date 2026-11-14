"""What the monitor collects. Each source has a tier, shown on every item:

    official    the institution itself (Fed, SEC, White House, BLS)
    mainstream  a primary newsroom (Reuters, Bloomberg, CNBC, MarketWatch)
    other       aggregators and fast wires that relay others (华尔街见闻 7x24, Yahoo per-ticker feeds)

Third-party text is sanitised and injection-flagged here, before it is stored or shown to a model.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

import requests

from equitydesk.policy.guards import sanitize_untrusted

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EquityDesk-monitor/1.0 (research tool)"}
TIMEOUT = 15
MAX_AGE_H = 48


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    tier: str                    # official | mainstream | other
    url: str
    kind: str = "rss"            # rss | wscn
    category: str = ""           # a hint for the scorer (policy, macro ...)
    params: dict = field(default_factory=dict, hash=False, compare=False)


def gnews(query: str) -> str:
    return f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"


SOURCES: tuple[Source, ...] = (
    Source("fed_monetary", "Federal Reserve · monetary policy", "official",
           "https://www.federalreserve.gov/feeds/press_monetary.xml", category="macro"),
    Source("fed_other", "Federal Reserve · press", "official", "https://www.federalreserve.gov/feeds/press_other.xml",
           category="macro"),
    Source("fed_speeches", "Federal Reserve · speeches", "official", "https://www.federalreserve.gov/feeds/speeches.xml",
           category="macro"),
    Source("sec", "SEC · press releases", "official", "https://www.sec.gov/news/pressreleases.rss", category="policy"),
    Source("whitehouse", "White House", "official", "https://www.whitehouse.gov/news/feed/", category="policy"),
    Source("bls", "BLS · latest numbers", "official", "https://www.bls.gov/feed/bls_latest.rss", category="macro"),
    Source("reuters", "Reuters", "mainstream", gnews("when:1d site:reuters.com (markets OR business OR economy)")),
    Source("bloomberg", "Bloomberg", "mainstream", gnews("when:1d site:bloomberg.com markets")),
    Source("cnbc_top", "CNBC Top", "mainstream",
           "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    Source("cnbc_econ", "CNBC Economy", "mainstream",
           "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258", category="macro"),
    Source("marketwatch", "MarketWatch", "mainstream", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    Source("wscn_us", "华尔街见闻 7x24 · 美股", "other",
           "https://api-one-wscn.awtmt.com/apiv1/content/lives?channel=us-stock-channel&limit=50", kind="wscn"),
    Source("wscn_global", "华尔街见闻 7x24 · 要闻", "other",
           "https://api-one-wscn.awtmt.com/apiv1/content/lives?channel=global-channel&limit=50", kind="wscn",
           params={"min_importance": 2}),
)
YAHOO = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"


def item_key(source_key: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source_key}|{url or title}".encode()).hexdigest()[:16]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    try:
        return _iso(parsedate_to_datetime(text))
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return _iso(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))
    except ValueError:
        return None


TAG = re.compile(r"<[^>]+>")
GNEWS_SUFFIX = re.compile(r"\s+-\s+[^-]{2,60}$")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_feed(xml: bytes, src: Source, hint_tickers: list[str] | None = None) -> list[dict]:
    """RSS 2.0 or Atom -> items. Google News titles lose their ' - Publisher' suffix."""
    root = ET.fromstring(xml)
    entries = [e for e in root.iter() if _local(e.tag) in ("item", "entry")]
    out, cutoff = [], datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_H)
    for e in entries:
        f = {_local(c.tag): c for c in e}
        title = (f["title"].text or "") if "title" in f else ""
        link = f.get("link")
        url = (link.text or link.get("href") or "").strip() if link is not None else ""
        when = None
        for k in ("pubDate", "published", "updated", "date"):
            if k in f:
                when = parse_time(f[k].text)
                if when:
                    break
        if not title.strip() or not when:
            continue
        if datetime.fromisoformat(when.replace("Z", "+00:00")) < cutoff:
            continue
        publisher = src.name
        if "news.google.com" in src.url:
            title = GNEWS_SUFFIX.sub("", title)
        desc = TAG.sub(" ", (f["description"].text or "") if "description" in f else "")
        t, flagged = sanitize_untrusted(title, max_len=300)
        s, flagged2 = sanitize_untrusted(desc, max_len=500)
        out.append({"key": item_key(src.key, url, t), "source_key": src.key, "source": publisher, "tier": src.tier,
                    "title": t, "url": url, "published": when, "summary": s, "flagged": flagged or flagged2,
                    "hint_tickers": hint_tickers or [], "category_hint": src.category})
    return out


def parse_wscn(data: dict, src: Source) -> list[dict]:
    """华尔街见闻 7x24 lives. Items often have no title: the first sentence of the text stands in."""
    out = []
    min_imp = int(src.params.get("min_importance", 0))
    for it in ((data or {}).get("data") or {}).get("items") or []:
        if int(it.get("score") or 1) < min_imp:
            continue
        text = (it.get("title") or "").strip() or re.split(r"[。！？]", it.get("content_text") or "")[0]
        if not text:
            continue
        t, flagged = sanitize_untrusted(text, max_len=200)
        s, _ = sanitize_untrusted(it.get("content_text") or "", max_len=500)
        when = datetime.fromtimestamp(int(it.get("display_time") or 0), tz=timezone.utc)
        url = it.get("uri") or ""
        out.append({"key": item_key(src.key, url, t), "source_key": src.key, "source": src.name, "tier": src.tier,
                    "title": t, "url": url, "published": _iso(when), "summary": s, "flagged": flagged,
                    "hint_tickers": [], "category_hint": src.category})
    return out


Getter = Callable[[str], requests.Response]


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def fetch(src: Source, get: Getter = _get) -> list[dict]:
    r = get(src.url)
    if src.kind == "wscn":
        r.encoding = "utf-8"
        return parse_wscn(r.json(), src)
    return parse_feed(r.content, src)


def yahoo_source(ticker: str) -> Source:
    return Source(f"yahoo_{ticker}", f"Yahoo {ticker}", "other", YAHOO.format(t=ticker))


def fetch_yahoo(ticker: str, get: Getter = _get) -> list[dict]:
    src = yahoo_source(ticker)
    return parse_feed(get(src.url).content, src, hint_tickers=[ticker])


def enabled(switches: dict[str, bool]) -> list[Source]:
    return [s for s in SOURCES if switches.get(s.key, True)]
