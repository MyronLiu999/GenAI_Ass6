"""Fundamental ratios from Yahoo's `info` and financial statements.

Statements are carried as {line_item: {period_iso: value}} so fixtures can be plain JSON.
"""

from __future__ import annotations

import math
from typing import Any


def num(x: Any) -> float | None:
    """A finite float, or None for missing / NaN / non-numeric values."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(x: Any) -> float | None:
    """A Yahoo fraction (0.253) as a percentage (25.3)."""
    v = num(x)
    return None if v is None else v * 100


def series(stmt: dict, item: str) -> list[tuple[str, float]]:
    """A statement line as [(period, value)] oldest first, numeric values only."""
    row = (stmt or {}).get(item) or {}
    pts = [(k, num(v)) for k, v in row.items()]
    return sorted([(k, v) for k, v in pts if v is not None])


def cagr_pct(stmt: dict, item: str, years: int = 3) -> float | None:
    pts = series(stmt, item)
    if len(pts) < 2:
        return None
    pts = pts[-(years + 1):]
    first, lastv = pts[0][1], pts[-1][1]
    n = len(pts) - 1
    if first <= 0 or lastv <= 0 or n <= 0:
        return None
    return ((lastv / first) ** (1 / n) - 1) * 100


def margin_change_pp(stmt: dict, numerator: str, denominator: str = "Total Revenue") -> float | None:
    """Change in a margin (percentage points) from the oldest to the newest period available."""
    a, b = dict(series(stmt, numerator)), dict(series(stmt, denominator))
    periods = sorted(set(a) & set(b))
    periods = [p for p in periods if b[p]]
    if len(periods) < 2:
        return None
    first, lastp = periods[0], periods[-1]
    return (a[lastp] / b[lastp] - a[first] / b[first]) * 100


def latest(stmt: dict, item: str) -> float | None:
    pts = series(stmt, item)
    return pts[-1][1] if pts else None
