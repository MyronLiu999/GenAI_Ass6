"""The whole desk, end to end, offline: scripted model + synthetic fixtures."""

import pytest

from equitydesk.harness.llm import ScriptedBackend
from equitydesk.harness.trace import read_trace
from equitydesk.policy.guards import InputError


def run(desk, ticker="DEMO", rounds=2):
    run_id = desk.start(ticker, rounds)
    events = list(desk.stream(run_id))
    return run_id, events


def test_full_run(make_desk):
    desk = make_desk()
    run_id, events = run(desk, rounds=2)
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert nodes[:2] == ["collect_data", "recall_memory"]
    assert set(nodes[2:7]) == {"fundamental", "technical", "sentiment", "risk", "web_research"}   # fan-out
    assert nodes[7:] == ["street", "bull", "bear", "bull", "bear", "pm", "grounding", "finalize"]
    done = events[-1]
    assert done["type"] == "done"
    rep = done["report"]
    assert rep["status"] == "ok" and rep["decision"]["rating"] == "HOLD" and rep["grounding"]["passed"]
    # web research: notes kept only with verified sources, and turned into W.* evidence
    research = rep["research"]
    assert research["status"] == "ok" and len(research["notes"]) == 2
    assert all(n["url"].startswith("https://example.com/") for n in research["notes"])
    assert {"W.1", "W.2"} <= {f["id"] for f in rep["facts"]}
    street = rep["analysts"]["street"]
    assert street["consensus_view"] and street["grounding"]["passed"]
    assert {h["id"] for h in rep["analysts"]["sentiment"]["headlines"]} == {"N.1", "N.2", "N.3", "N.4"}
    assert rep["decision"]["key_debates"] and rep["decision"]["variant_view"]
    assert len(rep["debate"]) == 4 and {t["side"] for t in rep["debate"]} == {"bull", "bear"}
    assert any("flagged" in w for w in rep["warnings"])                      # the planted injection headline
    assert "not investment advice" in rep["markdown"]
    assert desk.store.get_run(run_id)["status"] == "completed"
    trace = read_trace(run_id, desk.s.runs_dir)
    types = {e["type"] for e in trace}
    assert {"run_start", "tool_call", "node_start", "node_end", "llm_call", "grounding", "memory_read",
            "memory_write", "run_end"} <= types
    # 4 analysts + web search + note extraction + street analyst + 4 debate turns + PM
    assert sum(1 for e in trace if e["type"] == "llm_call") == 4 + 2 + 1 + 4 + 1
    assert any(e["type"] == "guard" and e.get("rule") == "provenance" for e in trace)


def test_web_research_off_still_runs_street_on_structured_data(make_desk):
    desk = make_desk()
    run_id = desk.start("DEMO", 1, web_research=False)
    rep = list(desk.stream(run_id))[-1]["report"]
    assert rep["research"]["status"] == "off" and not any(f["id"].startswith("W.") for f in rep["facts"])
    assert not rep["analysts"]["street"]["failed"]
    assert any(f["id"].startswith("A.") for f in rep["facts"])


def test_rounds_are_clamped(make_desk):
    desk = make_desk()
    _, events = run(desk, rounds=50)
    assert len(events[-1]["report"]["debate"]) == 2 * desk.s.graph.max_debate_rounds


def test_memory_carries_to_the_next_run(make_desk):
    desk = make_desk()
    desk.store.add_note("DEMO", "I hold 50 shares; never go above 5% of the portfolio.")
    run(desk, rounds=1)
    _, events = run(desk, rounds=1)
    mem = next(e for e in events if e["type"] == "node" and e["node"] == "recall_memory")["delta"]["memory"]
    assert len(mem["prior"]) == 1 and mem["prior"][0]["rating"] == "HOLD"
    assert "never go above 5%" in mem["text"] and "Your previous calls" in mem["text"]


def test_grounding_failure_triggers_one_revision(make_desk):
    desk = make_desk(ScriptedBackend(bad_citation_once=True))
    run_id, events = run(desk, rounds=1)
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert nodes[-5:] == ["pm", "grounding", "pm", "grounding", "finalize"]
    assert "street" in nodes
    g = events[-1]["report"]["grounding"]
    assert g["passed"] and g["revisions"] == 1
    prompts = [e for e in read_trace(run_id, desk.s.runs_dir) if e["type"] == "grounding" and e["node"] == "pm"]
    assert prompts[0]["unknown_ids"] == ["X.bogus"] and prompts[0]["retry"]


def test_revisions_are_bounded(make_desk, settings):
    s = settings.model_copy(update={"graph": settings.graph.model_copy(update={"max_revisions": 0})})
    desk = make_desk(ScriptedBackend(bad_citation_once=True), settings_=s)
    _, events = run(desk, rounds=1)
    g = events[-1]["report"]["grounding"]
    assert not g["passed"] and g["revisions"] == 0
    assert "FAILED" in events[-1]["report"]["markdown"]


def test_failed_analyst_degrades_not_crashes(make_desk, settings):
    class Flaky(ScriptedBackend):
        def structured(self, schema, messages):
            if "Technical Analyst" in messages[0].content:
                raise RuntimeError("bad request")       # not transient: no retry
            return super().structured(schema, messages)

    desk = make_desk(Flaky())
    _, events = run(desk, rounds=1)
    rep = events[-1]["report"]
    assert rep["analysts"]["technical"]["failed"] and rep["status"] == "ok"
    assert any("Technical Analyst failed" in w for w in rep["warnings"])


def test_budget_stop_then_resume(make_desk, settings):
    tight = settings.model_copy(update={"budget": settings.budget.model_copy(update={"run_token_ceiling": 3000})})
    desk = make_desk(settings_=tight)
    run_id, events = run(desk, rounds=1)
    assert events[-1]["type"] == "error" and events[-1]["kind"] == "budget_exceeded"
    assert desk.store.get_run(run_id)["status"] == "budget_exceeded"
    pending = desk.pending_nodes(run_id)
    assert pending, "the checkpoint keeps the run's place"
    desk2 = make_desk(settings_=settings)        # a new process with a normal budget, same stores
    events2 = list(desk2.stream(run_id, resume=True))
    assert events2[-1]["type"] == "done" and events2[-1]["report"]["grounding"]["passed"]
    assert "collect_data" not in [e["node"] for e in events2 if e["type"] == "node"]   # finished work is not redone
    assert desk2.store.get_run(run_id)["status"] == "completed"


def test_unknown_symbol_gives_no_data_report(make_desk):
    desk = make_desk()
    run_id, events = run(desk, ticker="ZZZZ")
    rep = events[-1]["report"]
    assert rep["status"] == "no_data" and "No usable price history" in rep["warnings"][0]
    assert desk.store.get_run(run_id)["status"] == "no_data"
    assert not any(e["type"] == "llm_call" for e in read_trace(run_id, desk.s.runs_dir))   # no tokens spent


def test_invalid_ticker_is_refused_before_anything_runs(make_desk):
    desk = make_desk()
    with pytest.raises(InputError):
        desk.start("DROP TABLE runs")
    assert desk.store.list_runs() == []


def test_follow_up_qa_keeps_its_thread(make_desk):
    from equitydesk.graph.qa import ReportQA
    desk = make_desk()
    run_id, events = run(desk, rounds=1)
    rep = events[-1]["report"]
    qa = ReportQA(desk)
    a1 = qa.ask(run_id, rep, "What is the stop?")
    assert "[" in a1
    qa.ask(run_id, rep, "And the RSI?")
    hist = ReportQA(make_desk()).history(run_id, rep)       # a fresh instance reads it from the checkpoint
    assert [r for r, _ in hist] == ["user", "assistant", "user", "assistant"]
