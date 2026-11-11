"""Follow-up Q&A on a finished report: a small LangGraph over MessagesState.

Each report gets its own conversation thread (thread_id "qa:<run_id>") in the same SQLite checkpointer,
so the conversation survives a page reload or a restart. The harness trims what the model sees to the last
`HISTORY_TURNS` messages; the full history stays in the checkpoint.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from equitydesk.graph.desk import prompt
from equitydesk.harness.llm import LLM
from equitydesk.harness.trace import Tracer
from equitydesk.tools.evidence import render

HISTORY_TURNS = 12


class ReportQA:
    def __init__(self, desk):
        self.desk = desk

    def _graph(self, run_id: str, report: dict):
        s = self.desk.s
        tracer = Tracer(run_id, s.runs_dir)
        llm = LLM(s, tracer, backend_factory=self.desk.factory, token_ceiling=s.budget.qa_token_ceiling)
        md = report.get("markdown", "")
        body = md.split("\n## Evidence", 1)[0]  # the evidence goes in separately, rendered for citation
        system = prompt("qa.md", ticker=report["ticker"], company=(report.get("company") or {}).get("name", ""),
                        language=report.get("language", "English"), report=body,
                        evidence=render(report.get("facts") or []))

        def answer(state: MessagesState) -> dict:
            with tracer.span("qa"):
                text = llm.chat("qa", "qa", system, state["messages"][-HISTORY_TURNS:])
            return {"messages": [AIMessage(text)]}

        g = StateGraph(MessagesState)
        g.add_node("answer", answer)
        g.add_edge(START, "answer")
        g.add_edge("answer", END)
        return g.compile(checkpointer=self.desk.saver), llm

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": f"qa:{run_id}"}}

    def ask(self, run_id: str, report: dict, question: str) -> str:
        graph, llm = self._graph(run_id, report)
        out = graph.invoke({"messages": [HumanMessage(question.strip()[:2000])]}, self.config(run_id))
        self.desk.store.add_usage(run_id, llm.meter.tokens, llm.meter.cost_usd)
        return out["messages"][-1].content

    def history(self, run_id: str, report: dict) -> list[tuple[str, str]]:
        graph, _ = self._graph(run_id, report)
        msgs = graph.get_state(self.config(run_id)).values.get("messages", [])
        return [("user" if isinstance(m, HumanMessage) else "assistant", m.content) for m in msgs]
