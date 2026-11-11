"""The research desk as a LangGraph StateGraph.

    START -> collect_data --(no data)--------------------------------------------------> finalize -> END
                  |
                  v
            recall_memory -> fundamental ┐
                          -> technical   ├─(join)-> bull -> bear --(round < N)--> bull
                          -> sentiment   │                    |
                          -> risk        ┘                    v (round == N)
                                                             pm -> grounding --(failed, revisions left)--> pm
                                                                       |
                                                                       v
                                                                    finalize -> END

Nodes are closures over a Runtime (settings, data provider, metered LLM, tracer, store), so nothing
unserialisable ever enters the checkpointed state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from string import Template
from typing import Callable

import yaml
from langgraph.graph import END, START, StateGraph

from equitydesk.config import Settings
from equitydesk.graph.state import DeskState
from equitydesk.harness.llm import LLM, BudgetExceeded, ConfigError, ModelError
from equitydesk.harness.trace import Tracer
from equitydesk.memory.store import Store
from equitydesk.memory.track import outcome, recall_text
from equitydesk.policy.grounding import (check_analyst, check_decision, consistency_warnings, normalize_point,
                                        normalize_text)
from equitydesk.policy.guards import DISCLAIMER
from equitydesk.report import build_plan, render_markdown
from equitydesk.schemas import AnalystReport, DebateTurn, Decision
from equitydesk.tools.evidence import Fact, build_facts, index, render
from equitydesk.tools.market import Provider

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
ANALYSTS = ("fundamental", "technical", "sentiment", "risk")


def prompt(name: str, **values) -> str:
    return Template((PROMPTS / name).read_text(encoding="utf-8")).safe_substitute(**values)


def roles() -> dict:
    return yaml.safe_load((PROMPTS / "roles.yaml").read_text(encoding="utf-8"))


@dataclass
class Runtime:
    settings: Settings
    provider: Provider
    llm: LLM
    tracer: Tracer
    store: Store


def render_analysts(reports: dict[str, dict]) -> str:
    spec = roles()
    out = ["<analyst_reports>"]
    for name in ANALYSTS:
        r = reports.get(name)
        if not r:
            continue
        title = spec[name]["title"]
        if r.get("failed"):
            out.append(f"### {title}: FAILED ({r.get('error', '')[:120]}). Treat this area as missing evidence.")
            continue
        out.append(f"### {title}: {r['stance']} (score {r['score']:+.2f}, confidence {r['confidence']})")
        out.append(r["summary"])
        out += [f"- {p['claim']}" for p in r.get("points", [])]
        if r.get("data_gaps"):
            out.append("Data gaps: " + "; ".join(r["data_gaps"]))
    out.append("</analyst_reports>")
    return "\n".join(out)


def render_debate(turns: list[dict]) -> str:
    if not turns:
        return "<debate>No arguments yet.</debate>"
    out = ["<debate>"]
    for t in turns:
        out.append(f"### Round {t['round']} — {t['side'].upper()}")
        out.append(t["argument"])
        if t.get("rebuttal"):
            out.append(f"Rebuttal: {t['rebuttal']}")
    out.append("</debate>")
    return "\n".join(out)


def build_graph(rt: Runtime) -> StateGraph:
    s, llm, tracer = rt.settings, rt.llm, rt.tracer
    spec = roles()

    def tool_hook(**kw):
        tracer.event("tool_call", **kw)

    def company_name(state: DeskState) -> str:
        return (state.get("company") or {}).get("name") or state["ticker"]

    # -- collect_data ----------------------------------------------------------------------------
    def collect_data(state: DeskState) -> dict:
        ticker = state["ticker"]
        with tracer.span("collect_data", ticker=ticker, provider=rt.provider.name):
            b = rt.provider.fetch(ticker, on_call=tool_hook)
            warnings = [f"Data component '{k}' unavailable ({v[:120]}); the desk worked without it."
                        for k, v in b.errors.items() if k not in ("history",)]
            if not b.has_prices:
                tracer.event("guard", rule="no_price_history", ticker=ticker, errors=b.errors)
                return {"status": "no_data", "facts": [], "company": {"name": ticker}, "data_source": b.source,
                        "warnings": [f"No usable price history for {ticker} "
                                     f"({b.errors.get('history', 'empty response')}). Check the symbol."]}
            facts = build_facts(b, state.get("profile") or {}, s)
            info = b.info or {}
            as_of = b.history.index[-1].date()
            stale = (date.today() - as_of).days
            if stale > s.data.staleness_days:
                warnings.append(f"Latest price is {stale} days old (as of {as_of}); figures may be stale.")
            flagged = [f.id for f in facts if f.flagged]
            if flagged:
                warnings.append(f"{len(flagged)} headline(s) contained instruction-like text and were flagged: "
                                + ", ".join(flagged))
                tracer.event("guard", rule="untrusted_text_flagged", ids=flagged)
            for f in facts:
                if f.id in ("S.consensus_check", "S.event_risk"):
                    warnings.append(f"{f.label}: {f.value}.")
                    tracer.event("guard", rule=f.id, detail=f.value)
            if info.get("quoteType") and info["quoteType"] != "EQUITY":
                warnings.append(f"{ticker} is a {info['quoteType']}, not a single stock: fundamental fields are "
                                "limited and the fundamental view should carry little weight.")
            bench = b.benchmark["Close"].dropna() if not b.benchmark.empty else None
            return {
                "status": "ok", "facts": [f.model_dump() for f in facts], "warnings": warnings,
                "data_source": b.source, "as_of": as_of.isoformat(),
                "price": float(b.history["Close"].iloc[-1]),
                "benchmark_price": float(bench.iloc[-1]) if bench is not None and len(bench) else None,
                "company": {"name": info.get("longName") or info.get("shortName") or ticker,
                            "sector": info.get("sector"), "industry": info.get("industry"),
                            "currency": info.get("currency") or "USD", "quote_type": info.get("quoteType")},
            }

    # -- recall_memory ---------------------------------------------------------------------------
    def recall_memory(state: DeskState) -> dict:
        with tracer.span("recall_memory"):
            prior = rt.store.list_analyses(state["ticker"], limit=3)
            for c in prior:
                c["outcome"] = outcome(c, state.get("price"), state.get("benchmark_price"), s.grounding.hold_band_pct,
                                       s.grounding.min_days_to_score)
            notes = rt.store.list_notes(state["ticker"])
            tracer.event("memory_read", prior_calls=len(prior), notes=len(notes))
            return {"memory": {"prior": prior, "notes": notes, "text": recall_text(prior, notes)}}

    # -- the four analysts -----------------------------------------------------------------------
    def analyst(name: str) -> Callable[[DeskState], dict]:
        role = spec[name]

        def node(state: DeskState) -> dict:
            with tracer.span(name):
                facts = state["facts"]
                system = prompt("analyst.md", title=role["title"], ticker=state["ticker"],
                                company=company_name(state), language=state.get("language", "English"),
                                focus=role["focus"])
                user = (render(facts, tuple(role["evidence"]), tuple(role.get("extra") or ()))
                        + "\n\nReturn your AnalystReport.")
                try:
                    rep = LLM.dump(llm.structured(name, "analyst", AnalystReport, system, user))
                    rep["points"] = [normalize_point(p) for p in rep["points"]]
                    rep["summary"] = normalize_text(rep["summary"])
                    rep["failed"] = False
                except (BudgetExceeded, ConfigError):
                    raise
                except ModelError as e:
                    rep = {"failed": True, "error": str(e)[:300], "stance": "neutral", "score": 0.0,
                           "confidence": 0, "summary": "", "points": [], "data_gaps": [f"{role['title']} failed"]}
                g = check_analyst(rep, index(facts), s.grounding) if not rep["failed"] else None
                rep["grounding"] = g.to_dict() if g else None
                warn = []
                if rep["failed"]:
                    warn.append(f"{role['title']} failed and was left out: {rep['error'][:160]}")
                elif not g.passed:
                    warn.append(f"{role['title']}: grounding issues ({len(g.unknown_ids)} unknown ids, "
                                f"{len(g.uncited_points)} uncited points, {len(g.ungrounded_numbers)} unmatched numbers).")
                tracer.event("grounding", node=name, **(g.to_dict() if g else {"passed": False, "failed": True}))
                return {"analyst_reports": {name: rep}, "warnings": warn}

        return node

    # -- bull / bear debate ----------------------------------------------------------------------
    def debater(side: str) -> Callable[[DeskState], dict]:
        def node(state: DeskState) -> dict:
            rnd = state.get("round", 0) + (1 if side == "bull" else 0)
            with tracer.span(side, round=rnd):
                system = prompt("debate.md", side=side.upper(), ticker=state["ticker"], company=company_name(state),
                                round=rnd, rounds=state["debate_rounds"], language=state.get("language", "English"))
                user = "\n\n".join([render_analysts(state.get("analyst_reports", {})),
                                    render_debate(state.get("debate", [])), render(state["facts"]),
                                    f"Return your DebateTurn for the {side.upper()} side."])
                turn = LLM.dump(llm.structured(side, "debate", DebateTurn, system, user))
                turn["points"] = [normalize_point(p) for p in turn["points"]]
                turn["argument"], turn["rebuttal"] = normalize_text(turn["argument"]), normalize_text(turn["rebuttal"])
                out: dict = {"debate": [{"side": side, "round": rnd, **turn}]}
                if side == "bull":
                    out["round"] = rnd
                return out

        return node

    def after_bear(state: DeskState) -> str:
        return "bull" if state.get("round", 0) < state["debate_rounds"] else "pm"

    # -- portfolio manager + grounding -----------------------------------------------------------
    def pm(state: DeskState) -> dict:
        revision = state.get("revisions", 0)
        with tracer.span("pm", revision=revision):
            p = state.get("profile") or {}
            profile = (f"<profile>risk tolerance: {p.get('risk_tolerance')}; preferred horizon: "
                       f"{p.get('horizon_months')} months; portfolio: ${float(p.get('portfolio_value', 0)):,.0f}; "
                       f"risk per trade: {p.get('risk_per_trade_pct')}%; max position: {p.get('max_position_pct')}%"
                       "</profile>")
            parts = [profile, f"<memory>\n{(state.get('memory') or {}).get('text', 'No memory.')}\n</memory>",
                     render_analysts(state.get("analyst_reports", {})), render_debate(state.get("debate", [])),
                     render(state["facts"])]
            g = state.get("grounding") or {}
            if revision and g.get("feedback"):
                parts.append("<grounding_feedback>\nYour previous Decision failed the code-run grounding check:\n"
                             f"{g['feedback']}\nPrevious Decision:\n{json.dumps(state.get('decision'), ensure_ascii=False)}\n"
                             "Return a corrected Decision.\n</grounding_feedback>")
            parts.append("Return your Decision.")
            system = prompt("pm.md", ticker=state["ticker"], company=company_name(state),
                            language=state.get("language", "English"))
            decision = LLM.dump(llm.structured("pm", "pm", Decision, system, "\n\n".join(parts)))
            for k in ("drivers", "risks"):
                decision[k] = [normalize_point(p) for p in decision[k]]
            for k in ("thesis", "memory_note", "override_rationale"):
                decision[k] = normalize_text(decision[k])
            decision["invalidation"] = [normalize_text(x) for x in decision["invalidation"]]
            return {"decision": decision}

    def grounding(state: DeskState) -> dict:
        with tracer.span("grounding"):
            g = check_decision(state["decision"], index(state["facts"]), s.grounding)
            revisions = state.get("revisions", 0)
            retry = (not g.passed) and revisions < s.graph.max_revisions
            rec = {**g.to_dict(), "feedback": g.feedback(), "retry": retry, "revisions": revisions}
            tracer.event("grounding", node="pm", **rec)
            out: dict = {"grounding": rec,
                         "consistency": consistency_warnings(state["decision"], state.get("analyst_reports", {}))}
            if retry:
                out["revisions"] = revisions + 1
            return out

    def after_grounding(state: DeskState) -> str:
        return "pm" if state["grounding"].get("retry") else "finalize"

    # -- finalize --------------------------------------------------------------------------------
    def finalize(state: DeskState) -> dict:
        with tracer.span("finalize"):
            base = {"run_id": state["run_id"], "ticker": state["ticker"], "company": state.get("company", {}),
                    "status": state.get("status", "ok"), "as_of": state.get("as_of"), "price": state.get("price"),
                    "benchmark": s.data.benchmark, "benchmark_price": state.get("benchmark_price"),
                    "data_source": state.get("data_source"), "warnings": state.get("warnings", []),
                    "disclaimer": DISCLAIMER, "generated_at": date.today().isoformat(),
                    "model": {r: s.model_for(r).name for r in ("analyst", "debate", "pm")},
                    "language": state.get("language", "English"), "facts": state.get("facts", [])}
            if base["status"] != "ok":
                base["markdown"] = render_markdown(base)
                return {"report": base}
            idx = index(state["facts"])
            decision = state["decision"]
            g = state.get("grounding") or {}
            report = {**base, "decision": decision, "plan": build_plan(decision, idx, state.get("profile") or {}),
                      "grounding": g, "consistency": state.get("consistency", []),
                      "analysts": state.get("analyst_reports", {}), "debate": state.get("debate", []),
                      "memory": {k: v for k, v in (state.get("memory") or {}).items() if k != "text"},
                      "usage": llm.meter.snapshot()}
            report["markdown"] = render_markdown(report)
            rt.store.save_analysis(state["run_id"], state["ticker"], state.get("as_of", ""), decision,
                                   state["price"], s.data.benchmark, state.get("benchmark_price"),
                                   bool(g.get("passed")), report)
            tracer.event("memory_write", kind="analysis", rating=decision["rating"])
            return {"report": report}

    # -- wiring ----------------------------------------------------------------------------------
    g = StateGraph(DeskState)
    g.add_node("collect_data", collect_data)
    g.add_node("recall_memory", recall_memory)
    for name in ANALYSTS:
        g.add_node(name, analyst(name))
    g.add_node("bull", debater("bull"))
    g.add_node("bear", debater("bear"))
    g.add_node("pm", pm)
    g.add_node("grounding", grounding)
    g.add_node("finalize", finalize)

    g.add_edge(START, "collect_data")
    g.add_conditional_edges("collect_data", lambda st: "recall_memory" if st.get("status") == "ok" else "finalize",
                            ["recall_memory", "finalize"])
    for name in ANALYSTS:
        g.add_edge("recall_memory", name)
    g.add_edge(list(ANALYSTS), "bull")
    g.add_edge("bull", "bear")
    g.add_conditional_edges("bear", after_bear, ["bull", "pm"])
    g.add_edge("pm", "grounding")
    g.add_conditional_edges("grounding", after_grounding, ["pm", "finalize"])
    g.add_edge("finalize", END)
    return g
