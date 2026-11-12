"""The earnings set-up and the quality of reported earnings, as cited facts. All arithmetic is here.

E.* — expectations: what the street expects for the quarter being reported next and for next year, set
      against what the company has actually been delivering (the sequential run-rate, the beat record).
      The comparison is the point: a consensus that asks for less than the run-rate is a low bar.
Q.* — reported quarterly results: the growth path (accelerating or decelerating), margin direction, and
      earnings-quality checks (receivables and inventory growing faster than sales, cash conversion,
      non-operating gains inside net income).

These are the fine-grained facts a sharp analyst argues from. Every figure is computed from reported or
consensus data; the scenarios (`E.q_rev_at_run_rate`) are labelled as arithmetic, not forecasts.
"""

from __future__ import annotations

from typing import Any

from equitydesk.tools.evidence import Fact, _Pack

DAYS_IN_QUARTER = 91
BAR_BAND_PP = 2.0          # implied growth within this many points of the run-rate = "in line with"
SEASONAL_GAP_PP = 8.0     # same-quarter-last-year growth this far from the recent average = seasonal business
UNEVEN_RANGE_PP = 25.0     # sequential growth spanning more than this over 4 quarters has no trend to read
FLAG_GAP_PP = 15.0         # receivables/inventory growth this far above revenue growth = quality flag


def _pct(a: float | None, b: float | None) -> float | None:
    return (a / b - 1) * 100 if a is not None and b not in (None, 0) else None


def _b(x: float | None) -> float | None:
    return x / 1e9 if x is not None else None


def earnings_facts(earn: dict, price: float | None, as_of: str, source: str = "Yahoo Finance",
                   sector: str | None = None) -> list[Fact]:
    p = _Pack(as_of, source)
    q = [x for x in (earn or {}).get("quarters") or [] if x.get("revenue")]
    if sector == "Financial Services":   # receivables, inventory and FCF say nothing useful about a bank
        q = [{k: v for k, v in x.items() if k not in ("receivables", "inventory", "fcf", "capex")} for x in q]
    _quarterly(p, q)
    _expectations(p, earn or {}, q, price)
    return p.facts


def _quarterly(p: _Pack, q: list[dict]) -> None:
    if not q:
        return
    last = q[0]
    per = last["period"]
    rev = [x["revenue"] for x in q]
    p.add("Q.rev_last", f"Revenue, quarter ended {per}", _b(rev[0]), "$B", as_of=per)
    qoq = [_pct(rev[i], rev[i + 1]) for i in range(min(4, len(rev) - 1))]
    if qoq:
        p.add("Q.rev_qoq_last", "Sequential (QoQ) revenue growth, last quarter", qoq[0], "%", as_of=per)
    if len(rev) >= 5:
        p.add("Q.rev_yoy_last", "Year-over-year revenue growth, last quarter", _pct(rev[0], rev[4]), "%", as_of=per)
    if len(qoq) >= 2:
        avg = sum(qoq) / len(qoq)
        p.add("Q.rev_qoq_avg", f"Average sequential revenue growth, last {len(qoq)} quarters", avg, "%", as_of=per)
        path = " → ".join(f"{x:.1f}%" for x in reversed(qoq))
        earlier = sum(qoq[1:]) / len(qoq[1:])
        uneven = max(qoq) - min(qoq) > UNEVEN_RANGE_PP     # seasonal or price-driven: no trend to read
        shape = ("uneven: seasonal or price-driven, compare year over year" if uneven else
                 "decelerating" if qoq[0] < earlier - BAR_BAND_PP else
                 "accelerating" if qoq[0] > earlier + BAR_BAND_PP else "steady")
        p.add("Q.rev_qoq_path", f"Sequential revenue growth, last {len(qoq)} quarters (oldest → newest)",
              f"{path} ({shape})", "text", as_of=per)

    def margin(x, key):
        return x[key] / x["revenue"] * 100 if x.get(key) is not None and x.get("revenue") else None

    base = q[min(4, len(q) - 1)]
    n_back = min(4, len(q) - 1)
    for key, fid, label in (("gross_profit", "gm", "Gross margin"), ("operating_income", "om", "Operating margin")):
        now = margin(last, key)
        p.add(f"Q.{fid}_last", f"{label}, last quarter", now, "%", as_of=per)
        then = margin(base, key)
        if now is not None and then is not None and n_back:
            p.add(f"Q.{fid}_change", f"{label} change over {n_back} quarters (percentage points)", now - then, "pp",
                  as_of=per)

    # earnings quality: are receivables / inventory running ahead of sales?
    if last.get("receivables") is not None:
        dso = last["receivables"] / last["revenue"] * DAYS_IN_QUARTER
        p.add("Q.dso_last", "Days sales outstanding (receivables / quarterly revenue x 91)", dso, "days", as_of=per)
        if base.get("receivables") is not None and n_back:
            p.add("Q.dso_then", f"Days sales outstanding {n_back} quarters earlier",
                  base["receivables"] / base["revenue"] * DAYS_IN_QUARTER, "days", as_of=base["period"])
    flags = []
    if len(q) >= 2 and qoq:
        # The flag compares year over year when a year-ago quarter exists: receivables and inventory are seasonal
        # too (a launch quarter builds both), so a sequential jump alone is not a warning.
        yoy_rev = _pct(rev[0], rev[4]) if len(rev) >= 5 else None
        for key, word in (("receivables", "receivables"), ("inventory", "inventory")):
            g = _pct(last.get(key), q[1].get(key))
            if g is None:
                continue
            p.add(f"Q.{key}_qoq", f"Sequential growth in {word}", g, "%", as_of=per)
            gy = _pct(last.get(key), q[4].get(key)) if len(q) >= 5 else None
            if gy is not None and yoy_rev is not None:
                p.add(f"Q.{key}_yoy", f"Year-over-year growth in {word}", gy, "%", as_of=per)
                if gy - yoy_rev >= FLAG_GAP_PP:
                    flags.append(f"{word} grew {gy:.1f}% year over year against revenue growth of {yoy_rev:.1f}% "
                                 f"(sequentially {g:.1f}% against {qoq[0]:.1f}%)")
            elif g - qoq[0] >= FLAG_GAP_PP:
                flags.append(f"{word} grew {g:.1f}% QoQ against revenue growth of {qoq[0]:.1f}%")
    if last.get("inventory") is not None and last.get("gross_profit") is not None:
        cogs = last["revenue"] - last["gross_profit"]
        if cogs > 0:
            p.add("Q.dio_last", "Days of inventory (inventory / quarterly cost of revenue x 91)",
                  last["inventory"] / cogs * DAYS_IN_QUARTER, "days", as_of=per)
    ni, oi = last.get("net_income"), last.get("operating_income")
    if last.get("fcf") is not None and ni and ni > 0:
        conv = last["fcf"] / ni * 100
        p.add("Q.fcf_last", "Free cash flow, last quarter", _b(last["fcf"]), "$B", as_of=per)
        p.add("Q.fcf_conversion", "Free cash flow as % of net income, last quarter", conv, "%", as_of=per)
        if conv < 60:
            flags.append(f"free cash flow was only {conv:.0f}% of net income")
    if last.get("capex") is not None:
        p.add("Q.capex_last", "Capital expenditure, last quarter", _b(abs(last["capex"])), "$B", as_of=per)
    if ni and oi and oi > 0:
        ratio = ni / oi * 100
        p.add("Q.ni_to_oi", "Net income as % of operating income, last quarter", ratio, "%", as_of=per)
        if ratio > 100:
            flags.append(f"net income was {ratio:.0f}% of operating income, so non-operating gains (for example on "
                         "investments) lifted reported earnings")
    if flags:
        p.add("Q.quality_flags", "Earnings-quality flags (computed)", "; ".join(flags), "text", as_of=per)


def _row(est: dict, period: str) -> dict:
    return (est or {}).get(period) or {}


def _expectations(p: _Pack, earn: dict, q: list[dict], price: float | None) -> None:
    beats = (earn.get("beats") or [])[:8]
    if beats:
        k = sum(1 for b in beats if b["eps_actual"] >= b["eps_est"])
        p.add("E.beat_record", "EPS against the consensus, last reported quarters",
              f"beat in {k} of the last {len(beats)} quarters", "text")
        s4 = [b["surprise_pct"] for b in beats[:4] if b.get("surprise_pct") is not None]
        if s4:
            p.add("E.surprise_avg_4q", f"Average EPS surprise, last {len(s4)} quarters", sum(s4) / len(s4), "%")
            p.add("E.surprise_last", f"EPS surprise, quarter reported {beats[0]['date']}", s4[0], "%",
                  as_of=beats[0]["date"])

    rev, eps = earn.get("rev_est") or {}, earn.get("eps_est") or {}
    r0, e0 = _row(rev, "0q"), _row(eps, "0q")
    last_rev = q[0]["revenue"] if q else None
    # 0q must be the quarter after the last reported one: its year-ago revenue is the quarter 3 back.
    aligned = bool(r0.get("avg") and last_rev and len(q) >= 4 and r0.get("yearAgoRevenue")
                   and abs(r0["yearAgoRevenue"] / q[3]["revenue"] - 1) < 0.03)
    if r0.get("avg"):
        p.add("E.q_rev_cons", "Consensus revenue, quarter to be reported next", _b(r0["avg"]), "$B")
        if r0.get("low") and r0.get("high"):
            p.add("E.q_rev_low", "Lowest analyst revenue estimate, next quarter", _b(r0["low"]), "$B")
            p.add("E.q_rev_high", "Highest analyst revenue estimate, next quarter", _b(r0["high"]), "$B")
        if r0.get("numberOfAnalysts"):
            p.add("E.q_rev_analysts", "Analysts estimating next-quarter revenue", r0["numberOfAnalysts"], "count")
    if aligned:
        implied = _pct(r0["avg"], last_rev)
        p.add("E.q_rev_implied_qoq", "Sequential revenue growth the consensus implies for the next quarter",
              implied, "%")
        if r0.get("high"):
            p.add("E.q_rev_high_implied_qoq", "Sequential growth implied by the highest revenue estimate",
                  _pct(r0["high"], last_rev), "%")
        # Year over year is free of seasonality, so it decides whether the bar is high or low. The sequential
        # references (recent average, same quarter a year ago) are shown beside it; for a seasonal or
        # price-driven business the recent sequential average is a poor yardstick.
        yoy_implied = _pct(r0["avg"], r0["yearAgoRevenue"])
        p.add("E.q_rev_implied_yoy", "Year-over-year revenue growth the consensus implies for the next quarter",
              yoy_implied, "%")
        qoq = [_pct(q[i]["revenue"], q[i + 1]["revenue"]) for i in range(min(4, len(q) - 1))]
        same_q = _pct(q[3]["revenue"], q[4]["revenue"]) if len(q) >= 5 else None
        if same_q is not None:
            p.add("E.q_rev_same_q_last_year", "Sequential revenue growth in the same quarter a year ago", same_q, "%")
        avg = sum(qoq) / len(qoq) if len(qoq) >= 2 else None
        steady = avg is not None and (same_q is None or abs(same_q - avg) <= SEASONAL_GAP_PP)
        yoy_last = _pct(q[0]["revenue"], q[4]["revenue"]) if len(q) >= 5 else None
        seq = f"sequentially it asks for {implied:+.1f}%"
        if avg is not None:
            seq += f" against a {avg:.1f}% average of the last {len(qoq)} quarters"
        if same_q is not None:
            seq += f" and {same_q:+.1f}% in the same quarter a year ago"
        if yoy_last is not None and yoy_implied is not None:
            gap = yoy_implied - yoy_last
            word = ("a lower bar" if gap < -BAR_BAND_PP else "a higher bar" if gap > BAR_BAND_PP else "a similar bar")
            text = (f"the consensus implies {yoy_implied:+.1f}% year-over-year revenue growth after {yoy_last:+.1f}% "
                    f"last quarter ({gap:+.1f} points: {word}); {seq}")
        elif avg is not None:
            word = ("below" if implied < avg - BAR_BAND_PP else "above" if implied > avg + BAR_BAND_PP
                    else "in line with")
            text = f"the consensus asks for {implied:+.1f}% sequential revenue growth, {word} the recent average; {seq}"
        else:
            text = ""
        if text:
            if not steady and avg is not None:
                text += " (sequential growth swings with the season or with prices here, so read the year-over-year view)"
            p.add("E.q_bar", "The bar for the next report (computed)", text, "text")
        if steady:
            at_rate = last_rev * (1 + avg / 100)
            p.add("E.q_rev_at_run_rate", "Next-quarter revenue if sequential growth matched its recent average "
                                         "(arithmetic, not a forecast)", _b(at_rate), "$B")
            p.add("E.q_rev_run_rate_vs_cons", "That run-rate revenue against the consensus", _pct(at_rate, r0["avg"]),
                  "%")
        if yoy_last is not None:
            held = r0["yearAgoRevenue"] * (1 + yoy_last / 100)
            p.add("E.q_rev_yoy_held", "Next-quarter revenue if year-over-year growth held at last quarter's pace "
                                      "(arithmetic, not a forecast)", _b(held), "$B")
            p.add("E.q_rev_yoy_held_vs_cons", "That revenue against the consensus", _pct(held, r0["avg"]), "%")
    if e0.get("avg"):
        p.add("E.q_eps_cons", "Consensus EPS, quarter to be reported next", e0["avg"], "$")
        if e0.get("yearAgoEps"):
            p.add("E.q_eps_growth_yoy", "Year-over-year EPS growth the consensus implies, next quarter",
                  _pct(e0["avg"], e0["yearAgoEps"]), "%")
    r1q = _row(rev, "+1q")
    if r1q.get("avg") and r0.get("avg") and aligned:
        p.add("E.q2_rev_cons", "Consensus revenue, the quarter after next", _b(r1q["avg"]), "$B")
        p.add("E.q2_rev_implied_qoq", "Sequential growth the consensus implies for the quarter after next",
              _pct(r1q["avg"], r0["avg"]), "%")
    ry, ny = _row(rev, "0y"), _row(rev, "+1y")
    if ry.get("avg"):
        p.add("E.fy_rev_cons", "Consensus revenue, current fiscal year", _b(ry["avg"]), "$B")
    if ny.get("avg"):
        p.add("E.ny_rev_cons", "Consensus revenue, next fiscal year", _b(ny["avg"]), "$B")
        if ny.get("growth") is not None:
            p.add("E.ny_rev_growth", "Consensus revenue growth, next fiscal year", ny["growth"] * 100, "%")
        if ny.get("low") and ny.get("high"):
            p.add("E.ny_rev_low", "Lowest analyst revenue estimate, next fiscal year", _b(ny["low"]), "$B")
            p.add("E.ny_rev_high", "Highest analyst revenue estimate, next fiscal year", _b(ny["high"]), "$B")
            p.add("E.ny_rev_dispersion", "Spread of next-year revenue estimates ((high - low) / consensus)",
                  (ny["high"] - ny["low"]) / ny["avg"] * 100, "%")
        if last_rev:
            p.add("E.ny_rev_vs_run_rate", "Next-year consensus revenue against the annualised last quarter (x4)",
                  _pct(ny["avg"], last_rev * 4), "%")
    eny = _row(eps, "+1y")
    if eny.get("avg") and price and eny["avg"] > 0:
        p.add("E.pe_ny", "P/E on next-year consensus EPS", price / eny["avg"], "x")
        if eny.get("low") and eny.get("high"):
            p.add("E.ny_eps_dispersion", "Spread of next-year EPS estimates ((high - low) / consensus)",
                  (eny["high"] - eny["low"]) / eny["avg"] * 100, "%")


def summary(facts: list[dict] | list[Fact]) -> dict[str, Any]:
    """The few E/Q facts the report headline shows, by id."""
    want = ("E.q_bar", "E.beat_record", "Q.rev_qoq_path", "Q.quality_flags", "E.ny_rev_vs_run_rate")
    out = {}
    for f in facts:
        f = f if isinstance(f, Fact) else Fact(**f)
        if f.id in want:
            out[f.id] = f
    return out
