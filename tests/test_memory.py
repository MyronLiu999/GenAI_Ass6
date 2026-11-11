import pytest

from equitydesk.memory.store import DEFAULT_PROFILE, Store
from equitydesk.memory.track import outcome, recall_text, summary, track_record


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "desk.db")


def test_profile_defaults_and_validation(store):
    assert store.get_profile() == DEFAULT_PROFILE
    store.set_profile({"risk_per_trade_pct": 0.5})
    assert store.get_profile()["risk_per_trade_pct"] == 0.5
    with pytest.raises(ValueError):
        store.set_profile({"hack": 1})


def test_notes_are_scoped_and_soft_deleted(store):
    a = store.add_note("NVDA", "I already hold 200 shares at $95.")
    store.add_note("*", "No tobacco or weapons stocks.")
    store.add_note("AAPL", "irrelevant to NVDA")
    assert {n["text"][:2] for n in store.list_notes("NVDA")} == {"I ", "No"}
    store.deactivate_note(a)
    assert [n["ticker"] for n in store.list_notes("NVDA")] == ["*"]
    with pytest.raises(ValueError):
        store.add_note("NVDA", "   ")


def test_analysis_roundtrip_and_resume_replaces(store):
    d = {"rating": "BUY", "conviction": 70, "horizon_months": 6, "thesis": "t"}
    store.save_analysis("r1", "NVDA", "2026-11-10", d, 100.0, "SPY", 500.0, True, {"x": 1})
    store.save_analysis("r1", "NVDA", "2026-11-10", d, 101.0, "SPY", 500.0, True, {"x": 2})  # resumed run
    calls = store.list_analyses("NVDA")
    assert len(calls) == 1 and calls[0]["price"] == 101.0 and calls[0]["score"] == 1
    assert store.get_report("r1") == {"x": 2}


def test_run_ledger(store):
    store.start_run("r1", "NVDA", "gpt-4o", 2, "yfinance")
    store.finish_run("r1", "budget_exceeded", 1000, 0.01, "boom")
    store.add_usage("r1", 50, 0.001)
    r = store.get_run("r1")
    assert r["status"] == "budget_exceeded" and r["tokens"] == 1050 and r["cost_usd"] == pytest.approx(0.011)


@pytest.mark.parametrize("score,price_now,bench_now,verdict", [
    (1, 110, 505, "hit"),     # +10% vs +1%
    (1, 101, 525, "miss"),    # +1% vs +5%
    (-1, 95, 500, "hit"),     # -5% vs 0%
    (0, 102, 500, "hit"),     # inside the 5% hold band
    (0, 120, 500, "miss"),
])
def test_outcome_verdicts(score, price_now, bench_now, verdict):
    call = {"price": 100.0, "benchmark_price": 500.0, "score": score, "created_at": "2026-10-01T10:00:00"}
    assert outcome(call, price_now, bench_now, 5.0)["verdict"] == verdict


def test_young_calls_are_too_early_to_score():
    from datetime import datetime, timedelta
    call = {"price": 100.0, "benchmark_price": 500.0, "score": 0,
            "created_at": (datetime.now() - timedelta(minutes=15)).isoformat(timespec="seconds"),
            "rating": "HOLD", "conviction": 60, "ticker": "NVDA", "benchmark": "SPY"}
    o = outcome(call, 100.0, 500.0, 5.0, min_days=5)       # the live case: a 15-minute-old HOLD was called a hit
    assert o["verdict"] == "too early" and o["return_pct"] == 0.0
    call["outcome"] = o
    assert "TOO EARLY" in recall_text([call], []) and "HIT" not in recall_text([call], [])
    df = track_record([call], {"NVDA": 100.0}, {"SPY": 500.0}, 5.0, min_days=5)
    assert summary(df)["scored"] == 0


def test_recall_text_and_summary():
    call = {"ticker": "NVDA", "price": 100.0, "benchmark_price": 500.0, "score": 1, "rating": "BUY", "conviction": 70,
            "created_at": "2026-10-01T10:00:00", "thesis": "AI capex", "benchmark": "SPY"}
    call["outcome"] = outcome(call, 110, 505, 5.0)
    text = recall_text([call], [{"ticker": "*", "text": "max 5% per position"}])
    assert "BUY" in text and "HIT" in text and "max 5% per position" in text
    assert "No previous calls" in recall_text([], [])
    df = track_record([call], {"NVDA": 110}, {"SPY": 505}, 5.0)
    s = summary(df)
    assert s["hit_rate"] == 1.0 and s["avg_excess_buys"] == pytest.approx(9.0)
