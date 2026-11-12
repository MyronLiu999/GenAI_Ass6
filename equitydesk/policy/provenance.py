"""Provenance for web research: only sources the search tool actually cited survive.

Live finding (2026-11-11): asking gpt-4o for structured output *while* searching returned zero
url_citation annotations and invented "sources". So the search runs as plain text with citations,
each cited span is replaced by a marker [S<n>] bound to the real URL (mark_citations), a second
model call extracts notes that must name their markers, and verify_notes drops every note whose
markers do not resolve to a cited URL. A URL in the final report is therefore one the tool visited.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


# Ratings-table sites: real, but they carry a rating and a target, not the analyst's reasoning.
AGGREGATORS = {"stockanalysis.com", "marketbeat.com", "tipranks.com", "pricetargets.com", "benzinga.com",
               "zacks.com", "wallstreetzen.com", "marketscreener.com", "finviz.com", "stocktitan.net"}


def is_aggregator(dom: str) -> bool:
    return any(dom == a or dom.endswith("." + a) for a in AGGREGATORS)


def normalize_url(url: str) -> str:
    """Comparable form: lower-case host, no fragment, no tracking parameters, no trailing slash."""
    try:
        p = urlsplit((url or "").strip())
    except ValueError:
        return ""
    if p.scheme not in ("http", "https") or not p.netloc:
        return ""
    query = urlencode([(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")])
    return urlunsplit((p.scheme, p.netloc.lower(), p.path.rstrip("/") or "/", query, ""))


def domain(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def mark_citations(text: str, annotations: list[dict]) -> tuple[str, list[dict]]:
    """Replace each cited span with ' [S<n>]' (one id per distinct URL, numbered by first appearance)
    and return the marked text with the source table. Overlapping spans keep the first one."""
    sources: dict[str, dict] = {}
    spans = []
    for a in sorted(annotations, key=lambda a: (a["start"], a["end"])):
        url = normalize_url(a.get("url", ""))
        if not url:
            continue
        if url not in sources:
            sources[url] = {"id": f"S{len(sources) + 1}", "url": url, "title": (a.get("title") or "").strip()[:200],
                            "domain": domain(url)}
        spans.append((a["start"], a["end"], sources[url]["id"]))
    out, last_end, pieces = text, None, []
    for start, end, sid in sorted(spans, key=lambda s: s[0], reverse=True):
        if last_end is not None and end > last_end:   # overlaps a span already replaced
            continue
        pieces.append((start, end, sid))
        last_end = start
    for start, end, sid in pieces:                      # already in descending order
        out = out[:start].rstrip() + f" [{sid}]" + out[end:]
    out = re.sub(r"[ \t]+([.,;:])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, list(sources.values())


def verify_notes(notes: list[dict], sources: list[dict], max_notes: int) -> tuple[list[dict], list[dict]]:
    """Keep notes whose source markers resolve to cited URLs; attach those URLs; drop duplicates.
    Returns (kept, dropped)."""
    by_id = {s["id"]: s for s in sources}
    kept, dropped, seen = [], [], set()
    for n in notes:
        valid = [i for i in n.get("source_ids") or [] if i in by_id]
        if not valid:
            dropped.append({**n, "reason": "no cited source"})
            continue
        key = ((n.get("source") or "").strip().lower(), n.get("date", ""), n.get("price_target"),
               (n.get("metric") or "").strip().lower(), n.get("figure"))
        if key in seen:
            dropped.append({**n, "reason": "duplicate"})
            continue
        seen.add(key)
        first = by_id[valid[0]]
        kept.append({**n, "source_ids": valid, "url": first["url"], "title": first["title"], "domain": first["domain"],
                     "urls": [by_id[i]["url"] for i in valid],
                     "aggregator": all(is_aggregator(by_id[i]["domain"]) for i in valid)})
    kept.sort(key=lambda n: n.get("date", ""), reverse=True)
    return kept[:max_notes], dropped
