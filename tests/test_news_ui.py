"""News tracking, and the HTML safety of the UI components (third-party text is rendered as HTML)."""

import time

import pytest

from equitydesk.memory.store import Store
from equitydesk.tools.news import collect, is_about, name_keys, triage


@pytest.mark.parametrize("ticker,company,title,about", [
    ("NVDA", "NVIDIA Corporation", "Nvidia hits most extreme selling level since 2019", True),
    ("NVDA", "NVIDIA Corporation", "Chip stocks' slide nears $5 trillion as NVDA tumbles", True),
    ("NVDA", "NVIDIA Corporation", "Rivian's vans got a serious upgrade", False),
    ("AAPL", "Apple Inc.", "Apple users get urgent security warning over iPhone flaw", True),
    ("AAPL", "Apple Inc.", "What could lift Netflix stock from here?", False),
    ("XOM", "Exxon Mobil Corporation", "ExxonMobil raises dividend", True),        # prefix match on the name
    ("XOM", "Exxon Mobil Corporation", "Exxon Mobil beats on refining margins", True),
    ("JPM", "JPMorgan Chase & Co.", "JPMorgan's Dimon warns on credit", True),
    ("T", "AT&T Inc.", "Stocks rally as T-bills yield more", False),                 # 1-letter tickers need the name
    ("T", "AT&T Inc.", "AT&T adds 400,000 fiber subscribers", True),
])
def test_is_about(ticker, company, title, about):
    assert is_about(title, name_keys(ticker, company)) is about


def test_first_seen_is_kept_and_triage_is_stored(settings, tmp_path):
    from equitydesk.harness.llm import ScriptedBackend
    from equitydesk.tools.market import FixtureProvider
    store = Store(tmp_path / "d.db")
    rows, errors = collect(FixtureProvider(settings), store, ["DEMO", "NOPE"])
    assert not errors and len(rows) == 4
    first = {r["key"]: r["first_seen"] for r in rows}
    flagged = [r for r in rows if r["flagged"]]
    assert len(flagged) == 1                                           # the planted injection headline
    time.sleep(1.1)
    again, _ = collect(FixtureProvider(settings), store, ["DEMO"])
    assert {r["key"]: r["first_seen"] for r in again} == first        # seen before: not new again
    out = triage(settings, store, again, backend_factory=lambda cfg: ScriptedBackend())
    assert out["triaged"] == 4
    assert all(r["triaged_at"] for r in store.list_news(["DEMO"]))
    assert store.get_run(next(r["run_id"] for r in store.list_runs() if r["ticker"] == "NEWS"))["status"] == "completed"


def test_third_party_text_is_escaped_in_html():
    from equitydesk.ui.components import chip, link, news_row, rich
    evil = '<img src=x onerror="alert(1)"> Buy now [T.price]'
    out = rich(evil)
    assert "<img" not in out and "&lt;img" in out and '<span class="ed-cite">T.price</span>' in out
    assert link("javascript:alert(1)") == ""
    assert 'href="https://a.com/x"' in link("https://a.com/x?utm_source=openai")
    row = news_row("2026-11-11", "<script>x</script>", "javascript:alert(1)", "", chip("<b>NEW</b>", "new"))
    assert "<script>" not in row and "javascript:" not in row and "&lt;b&gt;NEW" in row


@pytest.mark.parametrize("text,label", [
    ("Baird: downgrade Outperform → Neutral; target $500 → $200", "downgrade"),
    ("Citigroup: maintains Buy; target $315 → $285", "target cut"),
    ("Beta: maintains Buy; target $1,100 → $1,250", "target raised"),
    ("Goldman Sachs: reiterates Buy; target $300", "reiterated"),
    ("Delta: initiation Buy; target $140", "initiation"),
])
def test_action_tone(text, label):
    from equitydesk.ui.components import action_tone
    assert action_tone(text)[0] == label


class _SkippingBackend:
    """Answers every headline except the last one on its first call, like a model that drops items."""

    name = "scripted"

    def __init__(self):
        from equitydesk.harness.llm import ScriptedBackend
        self.inner, self.calls = ScriptedBackend(), 0

    def structured(self, schema, messages):
        obj, err, usage = self.inner.structured(schema, messages)
        self.calls += 1
        if self.calls == 1 and obj is not None and len(obj.items) > 1:
            obj = schema(items=obj.items[:-1])
        return obj, err, usage


def test_triage_asks_again_for_skipped_headlines_and_batches(settings, tmp_path, monkeypatch):
    from equitydesk.tools import news as newslib
    from equitydesk.tools.market import FixtureProvider
    store = Store(tmp_path / "d.db")
    rows, _ = collect(FixtureProvider(settings), store, ["DEMO"])
    backend = _SkippingBackend()
    out = triage(settings, store, rows, backend_factory=lambda cfg: backend)
    assert out == {"triaged": 4, "missing": 0, "cost_usd": 0.0}
    assert backend.calls == 2                                          # one retry for the skipped headline
    assert all(r["triaged_at"] for r in store.list_news(["DEMO"]))
    monkeypatch.setattr(newslib, "BATCH", 3)                           # 4 headlines -> 2 batches
    rows2 = [{**r, "triaged_at": None} for r in rows]
    b2 = _SkippingBackend()
    assert triage(settings, store, rows2, backend_factory=lambda cfg: b2)["triaged"] == 4
    assert b2.calls == 3                                               # batch 1, its retry, batch 2


def test_publish_time_is_shown_and_date_only_rows_upgrade(tmp_path):
    from equitydesk.ui.components import when_text
    assert when_text("2026-11-11") == "2026-11-11"
    shown = when_text("2026-11-11T14:05:00Z")
    assert len(shown) == 16 and shown[10] == " " and shown[13] == ":"
    store = Store(tmp_path / "d.db")
    item = {"title": "X", "url": "https://a.com/1", "publisher": "P"}
    store.track_news("DEMO", [{**item, "published": "2026-11-11"}])
    rows = store.track_news("DEMO", [{**item, "published": "2026-11-11T14:05:00Z"}])
    assert rows[0]["published"] == "2026-11-11T14:05:00Z"
