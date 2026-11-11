"""Run lifecycle: validate -> record in the ledger -> stream the graph (checkpointed) -> close the ledger.

    desk = Desk()                                   # settings.yaml, live data, configured model
    run_id = desk.start("NVDA", debate_rounds=2)
    for event in desk.stream(run_id): ...          # {"type": "node" | "done" | "error", ...}
    desk.stream(run_id, resume=True)               # continue a failed / budget-stopped run from its checkpoint

CLI:  python -m equitydesk.runner NVDA [--rounds 2] [--offline] [--resume RUN_ID]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import uuid
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from equitydesk.config import Settings, load_env, load_settings
from equitydesk.graph.desk import Runtime, build_graph
from equitydesk.harness.llm import LLM, BackendFactory, BudgetExceeded, ConfigError, ModelError
from equitydesk.harness.trace import Tracer
from equitydesk.memory.store import Store
from equitydesk.policy.guards import InputError, normalize_ticker
from equitydesk.tools.market import Provider, make_provider

RECURSION_LIMIT = 80   # supersteps; the debate and revision loops are bounded well below this in code


def new_run_id(ticker: str) -> str:
    safe = "".join(ch for ch in ticker if ch.isalnum())[:8] or "X"
    return f"{safe}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


class Desk:
    def __init__(self, settings: Settings | None = None, *, store: Store | None = None,
                 provider: Provider | None = None, backend_factory: BackendFactory | None = None):
        load_env()
        self.s = settings or load_settings()
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or Store(self.s.store_path)
        self.provider = provider or make_provider(self.s)
        self.factory = backend_factory
        self._conn = sqlite3.connect(self.s.checkpoint_path, check_same_thread=False)
        self.saver = SqliteSaver(self._conn)
        self._pending: dict[str, dict] = {}   # registered by start(), consumed by stream()

    def close(self) -> None:
        self._conn.close()

    # -- graph per run -------------------------------------------------------------------------
    def _compiled(self, run_id: str):
        tracer = Tracer(run_id, self.s.runs_dir)
        llm = LLM(self.s, tracer, backend_factory=self.factory)
        rt = Runtime(settings=self.s, provider=self.provider, llm=llm, tracer=tracer, store=self.store)
        return build_graph(rt).compile(checkpointer=self.saver), rt

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self, ticker: str, debate_rounds: int | None = None, language: str | None = None,
              web_research: bool | None = None) -> str:
        """Validate and register a run. Raises InputError before anything is fetched or called."""
        t = normalize_ticker(ticker)
        rounds = self.s.clamp_rounds(debate_rounds)
        run_id = new_run_id(t)
        self._pending[run_id] = {"run_id": run_id, "ticker": t, "debate_rounds": rounds,
                         "language": language or self.s.graph.language, "profile": self.store.get_profile(),
                         "round": 0, "revisions": 0,
                         "use_web": self.s.research.web_search if web_research is None else bool(web_research)}
        self.store.start_run(run_id, t, self.s.model_for("pm").name, rounds, self.provider.name)
        return run_id

    def stream(self, run_id: str, resume: bool = False) -> Iterator[dict]:
        graph, rt = self._compiled(run_id)
        cfg = self.config(run_id)
        if resume:
            snap = graph.get_state(cfg)
            if not snap.next:
                yield {"type": "error", "kind": "not_resumable", "error": f"run {run_id} has nothing left to run"}
                return
            inp = None
            self.store.mark_running(run_id)
            rt.tracer.event("run_resume", next=list(snap.next))
        else:
            pending = self._pending.pop(run_id, None)
            if pending is None:
                raise ValueError("call start() first, or pass resume=True")
            inp = pending
            rt.tracer.event("run_start", ticker=pending["ticker"], debate_rounds=pending["debate_rounds"],
                            language=pending["language"], provider=self.provider.name,
                            models={r: self.s.model_for(r).name for r in ("analyst", "debate", "pm")})
        meter = rt.llm.meter
        try:
            for update in graph.stream(inp, cfg, stream_mode="updates"):
                for node, delta in update.items():
                    yield {"type": "node", "node": node, "delta": delta or {}, "usage": meter.snapshot()}
            final = graph.get_state(cfg).values
            report = final.get("report") or {}
            status = "completed" if report.get("status") == "ok" else "no_data"
            self.store.finish_run(run_id, status, meter.tokens, meter.cost_usd)
            rt.tracer.event("run_end", status=status, **meter.snapshot())
            yield {"type": "done", "run_id": run_id, "report": report, "usage": meter.snapshot()}
        except BudgetExceeded as e:
            yield self._fail(rt, run_id, "budget_exceeded", e)
        except ConfigError as e:
            yield self._fail(rt, run_id, "config_error", e)
        except ModelError as e:
            yield self._fail(rt, run_id, "model_error", e)
        except Exception as e:  # the run is checkpointed: it can be resumed after the cause is fixed
            yield self._fail(rt, run_id, "failed", e)

    def _fail(self, rt: Runtime, run_id: str, status: str, e: Exception) -> dict:
        msg = f"{type(e).__name__}: {e}"[:800]
        m = rt.llm.meter
        self.store.finish_run(run_id, status, m.tokens, m.cost_usd, msg)
        rt.tracer.event("run_end", status=status, error=msg, **m.snapshot())
        # Every node's output is checkpointed, so any failed run can be resumed once the cause is fixed.
        return {"type": "error", "kind": status, "run_id": run_id, "error": msg, "usage": m.snapshot(),
                "resumable": True}

    def analyze(self, ticker: str, debate_rounds: int | None = None, language: str | None = None,
                web_research: bool | None = None) -> dict:
        """Run to completion and return the report; raises RuntimeError if the run fails."""
        run_id = self.start(ticker, debate_rounds, language, web_research)
        last = None
        for ev in self.stream(run_id):
            last = ev
        if last is None or last["type"] != "done":
            raise RuntimeError((last or {}).get("error", "run produced no result"))
        return last["report"]

    def state(self, run_id: str) -> dict:
        graph, _ = self._compiled(run_id)
        return graph.get_state(self.config(run_id)).values

    def pending_nodes(self, run_id: str) -> list[str]:
        graph, _ = self._compiled(run_id)
        return list(graph.get_state(self.config(run_id)).next)


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m equitydesk.runner", description="Run the research desk headless.")
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--language", choices=["English", "Chinese"], default=None)
    ap.add_argument("--offline", action="store_true", help="scripted model + recorded fixtures (free)")
    ap.add_argument("--resume", metavar="RUN_ID")
    ap.add_argument("--no-web", action="store_true", help="skip web research (structured sell-side data only)")
    ap.add_argument("--out", help="write the Markdown report here")
    a = ap.parse_args(argv)
    overrides = {"model": {"provider": "scripted", "name": "scripted"}, "data": {"provider": "fixtures"}} if a.offline else {}
    desk = Desk(load_settings(overrides=overrides))
    try:
        run_id = a.resume or desk.start(a.ticker or "", a.rounds, a.language, False if a.no_web else None)
    except InputError as e:
        print(f"refused: {e}")
        return 2
    for ev in desk.stream(run_id, resume=bool(a.resume)):
        if ev["type"] == "node":
            u = ev["usage"]
            print(f"  {ev['node']:<14} done   tokens={u['tokens']:>7,}  cost=${u['cost_usd']:.4f}")
        elif ev["type"] == "error":
            print(f"RUN {ev.get('kind')}: {ev['error']}\nresume with: python -m equitydesk.runner --resume {run_id}")
            return 1
        else:
            rep = ev["report"]
            if a.out:
                with open(a.out, "w", encoding="utf-8") as f:
                    f.write(rep["markdown"])
            d = rep.get("decision") or {}
            print(f"\n{rep['ticker']}: {d.get('rating', 'NO REPORT')} conviction={d.get('conviction')} "
                  f"grounded={(rep.get('grounding') or {}).get('passed')}  run={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
