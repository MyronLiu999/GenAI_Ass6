import math

import numpy as np
import pandas as pd
import pytest

from equitydesk.tools import fundamentals as fx
from equitydesk.tools import indicators as ind
from equitydesk.tools import risk as rk
from equitydesk.tools.evidence import build_facts, fmt, index, render
from equitydesk.tools.market import FixtureProvider, normalize_news


def series(vals):
    return pd.Series(vals, index=pd.bdate_range("2025-01-01", periods=len(vals)), dtype=float)


def test_rsi_extremes():
    assert ind.last(ind.rsi(series(range(1, 40)))) == 100.0
    assert ind.last(ind.rsi(series([10.0] * 40))) == 50.0
    assert ind.last(ind.rsi(series(range(40, 1, -1)))) == pytest.approx(0.0, abs=1e-9)


def test_rsi_matches_wilder_reference():
    # The widely reproduced 14-day example (StockCharts' RSI page): first RSI 70.46, then Wilder-smoothed.
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
              46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    delta = pd.Series(closes).diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag, al = gain.iloc[1:15].mean(), loss.iloc[1:15].mean()
    expected = [100 - 100 / (1 + ag / al)]
    for i in range(15, len(closes)):
        ag, al = (ag * 13 + gain.iloc[i]) / 14, (al * 13 + loss.iloc[i]) / 14
        expected.append(100 - 100 / (1 + ag / al))
    assert expected[0] == pytest.approx(70.46, abs=0.05)
    got = ind.rsi(series(closes)).iloc[14:].tolist()
    assert got == pytest.approx(expected, abs=1e-9)
    assert ind.rsi(series(closes)).iloc[:14].isna().all()


def test_sma_returns_and_ytd():
    s = series(range(1, 301))
    assert ind.last(ind.sma(s, 20)) == pytest.approx(290.5)
    assert ind.pct_return(s, 10) == pytest.approx((300 / 290 - 1) * 100)
    assert ind.pct_return(series([1, 2]), 5) is None


def test_atr_constant_range():
    n = 50
    close = series([100.0] * n)
    assert ind.last(ind.atr(close + 1, close - 1, close)) == pytest.approx(2.0)


def test_drawdown_beta_var():
    s = series([100, 120, 90, 110])
    assert rk.max_drawdown_pct(s) == pytest.approx(-25.0)
    assert rk.current_drawdown_pct(s) == pytest.approx((110 / 120 - 1) * 100)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    assert rk.beta(r, r) == pytest.approx(1.0)
    assert rk.beta(2 * r, r) == pytest.approx(2.0)
    assert rk.var_hist_pct(r) > 0


def test_trade_plan_sizes_to_risk_budget():
    plan = rk.trade_plan(price=100, atr_value=2.5, portfolio_value=100_000, risk_per_trade_pct=1.0,
                         max_position_pct=50, atr_multiple=2.0)
    assert plan["stop"] == pytest.approx(95.0)
    assert plan["stop_pct"] == pytest.approx(5.0)
    assert plan["position_pct"] == pytest.approx(20.0)        # 1% risk / 5% stop
    assert plan["shares"] == 200
    assert plan["risk_usd"] == pytest.approx(1000.0)
    capped = rk.trade_plan(100, 2.5, 100_000, 1.0, max_position_pct=10)
    assert capped["position_pct"] == 10
    assert rk.trade_plan(100, None, 1, 1, 1) is None


def test_fundamentals_helpers():
    stmt = {"Total Revenue": {"2022": 100.0, "2023": 110.0, "2024": 121.0, "2025": 133.1},
            "Operating Income": {"2022": 10.0, "2025": 26.62}}
    assert fx.cagr_pct(stmt, "Total Revenue", 3) == pytest.approx(10.0)
    assert fx.margin_change_pp(stmt, "Operating Income") == pytest.approx(10.0)
    assert fx.num("x") is None and fx.num(float("nan")) is None
    assert fx.pct(0.25) == 25.0


def test_fmt():
    assert fmt(-37.8, "$B") == "-$37.8B"
    assert fmt(1234.5, "$") == "$1,234.50"
    assert fmt(-4.94, "%") == "-4.9%"
    assert fmt(40.0, "count") == "40"
    assert fmt(None, "%") == "n/a"


def test_news_normaliser_accepts_both_shapes():
    nested = [{"content": {"title": "A", "provider": {"displayName": "Reuters"}, "pubDate": "2026-11-01T10:00:00Z",
                           "canonicalUrl": {"url": "u"}}}]
    flat = [{"title": "B", "publisher": "WSJ", "providerPublishTime": 1762000000, "link": "v"}]
    out = normalize_news(nested + flat + [{"content": {"title": ""}}], limit=10)
    assert [(n["title"], n["publisher"], n["url"]) for n in out] == [("A", "Reuters", "u"), ("B", "WSJ", "v")]
    assert out[0]["published"] == "2026-11-01T10:00:00Z"
    assert out[1]["published"].startswith("2025-11-01T")


def test_evidence_pack(settings):
    b = FixtureProvider(settings).fetch("DEMO")
    facts = build_facts(b, {"portfolio_value": 100_000, "risk_per_trade_pct": 1, "max_position_pct": 10}, settings)
    idx = index(facts)
    for fid in ("T.price", "T.rsi14", "R.vol_1y", "R.beta_1y", "F.pe_ttm", "F.rev_cagr_3y", "S.target_upside",
                "P.stop", "P.position_pct", "N.1"):
        assert fid in idx, fid
    assert idx["T.price"].numeric == pytest.approx(float(b.history["Close"].iloc[-1]), rel=1e-6)
    assert len(idx) == len(facts), "fact ids are unique"
    flagged = [f for f in facts if f.flagged]
    assert [f.id for f in flagged] == ["N.3"]
    text = render(facts)
    assert "<untrusted" in text and "[N.3]" in text and "FLAGGED" in text
    assert text.index("</evidence>") < text.index("<untrusted")


def test_trend_regime_looks_at_price_not_just_the_cross():
    from equitydesk.tools.evidence import trend_regime
    # the live NVDA case: 50-day above 200-day, but the price 22% below the 50-day
    assert trend_regime(168.89, 217.97, 204.26).startswith("breakdown within a prior uptrend")
    assert trend_regime(250, 217.97, 204.26).startswith("uptrend")
    assert trend_regime(150, 180, 200).startswith("downtrend")
    assert trend_regime(190, 180, 200).startswith("transition")


def test_rsi_zone_and_stale_consensus():
    from equitydesk.tools.evidence import rsi_zone, stale_consensus
    assert rsi_zone(23.9).startswith("oversold") and "not a continuation" in rsi_zone(23.9)
    assert rsi_zone(75).startswith("overbought") and rsi_zone(50).startswith("neutral")
    assert stale_consensus(-25.4, 92.9)            # the live NVDA case
    assert not stale_consensus(-3.0, 92.9) and not stale_consensus(-25.4, 10.0) and not stale_consensus(None, 50)


def test_event_risk_fact_near_earnings(settings):
    from datetime import date, timedelta
    b = FixtureProvider(settings).fetch("DEMO")
    b.calendar = {"next_earnings": (date.today() + timedelta(days=6)).isoformat()}
    idx = index(build_facts(b, {}, settings))
    assert "S.event_risk" in idx and "6 days" in idx["S.event_risk"].value
    b.calendar = {"next_earnings": (date.today() + timedelta(days=60)).isoformat()}
    assert "S.event_risk" not in index(build_facts(b, {}, settings))


def test_decision_schema_requires_two_sided_invalidation_and_memory_note():
    from pydantic import ValidationError
    from equitydesk.schemas import Decision
    base = dict(rating="HOLD", conviction=50, horizon_months=6, thesis="t", drivers=[], risks=[],
                memory_note="No previous calls on this ticker.", key_debates=["Is growth durable?"],
                variant_view="In line with the street.")
    with pytest.raises(ValidationError):
        Decision(**base, invalidation=["only one side"])
    with pytest.raises(ValidationError):
        Decision(**{**base, "memory_note": ""}, invalidation=["up", "down"])
    assert Decision(**base, invalidation=["up", "down"])


def test_thin_history_degrades(settings):
    b = FixtureProvider(settings).fetch("FLAT")
    idx = index(build_facts(b, {}, settings))
    assert "T.price" in idx and "T.sma200" not in idx and "T.ret_1y" not in idx


def test_action_text_survives_pandas_nan():
    """Actions passed through a DataFrame turn a missing prior target into NaN."""
    import pandas as pd
    from equitydesk.tools.evidence import _action_text
    rec = pd.DataFrame([{"firm": "X", "action": "up", "from": "Hold", "to": "Buy", "pt": 425.0, "prior_pt": None},
                        {"firm": "Y", "action": "main", "from": None, "to": "Buy", "pt": None, "prior_pt": None}]
                       ).to_dict("records")
    assert _action_text(rec[0]) == "X: upgrade Hold → Buy; target $425"
    assert "nan" not in _action_text(rec[1]).lower()
