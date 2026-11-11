"""Reflection and the track record: how did past calls do against the benchmark?

A call is scored on excess return (stock minus benchmark) since the day it was made:
  BUY / STRONG_BUY   hit if excess > 0
  SELL / STRONG_SELL hit if excess < 0
  HOLD               hit if |excess| <= grounding.hold_band_pct
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def outcome(call: dict, price_now: float | None, bench_now: float | None, hold_band_pct: float,
            min_days: int = 0) -> dict:
    out = {"return_pct": None, "bench_pct": None, "excess_pct": None, "verdict": "n/a", "days": None}
    try:
        out["days"] = (datetime.now() - datetime.fromisoformat(call["created_at"])).days
    except (KeyError, ValueError):
        pass
    if not price_now or not call.get("price"):
        return out
    ret = (price_now / call["price"] - 1) * 100
    out["return_pct"] = ret
    if bench_now and call.get("benchmark_price"):
        b = (bench_now / call["benchmark_price"] - 1) * 100
        out["bench_pct"], out["excess_pct"] = b, ret - b
    if out["days"] is not None and out["days"] < min_days:
        out["verdict"] = "too early"     # a few days of noise is not evidence either way
        return out
    ex = out["excess_pct"] if out["excess_pct"] is not None else ret
    score = call.get("score", 0)
    if score > 0:
        out["verdict"] = "hit" if ex > 0 else "miss"
    elif score < 0:
        out["verdict"] = "hit" if ex < 0 else "miss"
    else:
        out["verdict"] = "hit" if abs(ex) <= hold_band_pct else "miss"
    return out


def recall_text(prior: list[dict], notes: list[dict]) -> str:
    """The memory block the PM reads: its own past calls with outcomes, then the user's notes."""
    lines = []
    if prior:
        lines.append("Your previous calls on this ticker (most recent first), scored against what happened since:")
        for c in prior:
            o = c.get("outcome") or {}
            perf = "no outcome yet"
            if o.get("return_pct") is not None:
                perf = f"stock {o['return_pct']:+.1f}%"
                if o.get("excess_pct") is not None:
                    perf += f", vs benchmark {o['excess_pct']:+.1f}%"
                perf += f" over {o.get('days', '?')} days -> {o['verdict'].upper()}"
                if o["verdict"] == "too early":
                    perf += " (no evidence either way yet)"
            lines.append(f"- {c['created_at'][:10]}: {c['rating']} (conviction {c.get('conviction')}) at "
                         f"${c['price']:.2f}; {perf}. Thesis then: {(c.get('thesis') or '')[:300]}")
    else:
        lines.append("No previous calls on this ticker.")
    if notes:
        lines.append("The user's notes (their own statements; respect their constraints):")
        lines += [f"- [{n['ticker']}] {n['text']}" for n in notes]
    return "\n".join(lines)


def track_record(calls: list[dict], prices_now: dict[str, float], bench_now: dict[str, float],
                 hold_band_pct: float, min_days: int = 0) -> pd.DataFrame:
    rows = []
    for c in calls:
        o = outcome(c, prices_now.get(c["ticker"]), bench_now.get(c.get("benchmark") or ""), hold_band_pct, min_days)
        rows.append({**{k: c.get(k) for k in ("created_at", "ticker", "rating", "score", "conviction", "horizon_months",
                                               "price", "grounded", "run_id")},
                     "price_now": prices_now.get(c["ticker"]), **o})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    scored = df[df["verdict"].isin(["hit", "miss"])] if not df.empty else df
    if scored.empty:
        return {"calls": int(len(df)), "scored": 0, "hit_rate": None, "avg_excess_buys": None, "avg_excess_sells": None}
    buys = scored[scored["score"] > 0]["excess_pct"].dropna()
    sells = scored[scored["score"] < 0]["excess_pct"].dropna()
    return {"calls": int(len(df)), "scored": int(len(scored)),
            "hit_rate": float((scored["verdict"] == "hit").mean()),
            "avg_excess_buys": float(buys.mean()) if not buys.empty else None,
            "avg_excess_sells": float(sells.mean()) if not sells.empty else None}
