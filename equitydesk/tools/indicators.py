"""Technical indicators as pure pandas functions. Wilder smoothing where the standard calls for it."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x: pd.Series, n: int, start: int = 0) -> pd.Series:
    """Wilder smoothing: seeded with the simple mean of the first n values from `start`, then
    avg = (avg * (n - 1) + x) / n. (pandas' ewm seeds differently and disagrees for many bars.)"""
    vals = x.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) < start + n:
        return pd.Series(out, index=x.index)
    avg = float(np.mean(vals[start:start + n]))
    out[start + n - 1] = avg
    for i in range(start + n, len(vals)):
        avg = (avg * (n - 1) + vals[i]) / n
        out[i] = avg
    return pd.Series(out, index=x.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. 100 when there were no losses in the window, 50 when the price never moved."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder(gain, n, start=1)   # the first diff is NaN
    avg_loss = wilder(loss, n, start=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + avg_gain / avg_loss)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = upper - lower
    pctb = (close - lower) / width.replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "pctb": pctb})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def pct_return(close: pd.Series, days: int) -> float | None:
    """Return over the last `days` trading days, in percent. None if the history is too short."""
    close = close.dropna()
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


def ytd_return(close: pd.Series) -> float | None:
    close = close.dropna()
    if close.empty:
        return None
    year = close.index[-1].year
    prior = close[close.index.year < year]
    if prior.empty:
        return None
    return float(close.iloc[-1] / prior.iloc[-1] - 1) * 100


def last(s: pd.Series) -> float | None:
    s = s.dropna()
    return None if s.empty else float(s.iloc[-1])
