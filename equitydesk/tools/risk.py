"""Risk metrics and the trade plan. Percent units throughout unless a name says otherwise."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.dropna().pct_change().dropna()


def ann_vol_pct(r: pd.Series) -> float | None:
    return None if len(r) < 20 else float(r.std(ddof=1) * math.sqrt(TRADING_DAYS) * 100)


def max_drawdown_pct(close: pd.Series) -> float | None:
    close = close.dropna()
    if close.empty:
        return None
    return float((close / close.cummax() - 1).min() * 100)


def current_drawdown_pct(close: pd.Series) -> float | None:
    close = close.dropna()
    if close.empty:
        return None
    return float((close.iloc[-1] / close.max() - 1) * 100)


def _aligned(r: pd.Series, rb: pd.Series) -> pd.DataFrame:
    return pd.concat([r.rename("a"), rb.rename("b")], axis=1, join="inner").dropna()


def beta(r: pd.Series, rb: pd.Series) -> float | None:
    df = _aligned(r, rb)
    if len(df) < 60 or df["b"].var() == 0:
        return None
    return float(df["a"].cov(df["b"]) / df["b"].var())


def correlation(r: pd.Series, rb: pd.Series) -> float | None:
    df = _aligned(r, rb)
    return None if len(df) < 60 else float(df["a"].corr(df["b"]))


def var_hist_pct(r: pd.Series, level: float = 0.95) -> float | None:
    """One-day historical VaR as a positive loss percentage."""
    return None if len(r) < 60 else float(-np.percentile(r, (1 - level) * 100) * 100)


def sharpe(r: pd.Series, rf_pct: float) -> float | None:
    if len(r) < 60 or r.std(ddof=1) == 0:
        return None
    excess = r.mean() * TRADING_DAYS - rf_pct / 100
    return float(excess / (r.std(ddof=1) * math.sqrt(TRADING_DAYS)))


def trade_plan(price: float, atr_value: float | None, portfolio_value: float, risk_per_trade_pct: float,
               max_position_pct: float, atr_multiple: float = 2.0) -> dict | None:
    """ATR stop and fixed-fractional position size.

    stop = price - atr_multiple * ATR. Size so that being stopped out loses risk_per_trade_pct of the
    portfolio, capped at max_position_pct of the portfolio.
    """
    if not price or not atr_value or price <= 0 or atr_value <= 0:
        return None
    stop = max(price - atr_multiple * atr_value, 0.01)
    stop_pct = (price - stop) / price * 100
    position_pct = min(max_position_pct, risk_per_trade_pct / stop_pct * 100)
    position_usd = portfolio_value * position_pct / 100
    shares = int(position_usd // price)
    return {
        "stop": stop,
        "stop_pct": stop_pct,
        "position_pct": position_pct,
        "shares": shares,
        "risk_usd": shares * (price - stop),
    }
