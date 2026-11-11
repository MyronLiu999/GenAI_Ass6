"""The desk's LangGraph state. Plain JSON-able values only, so every checkpoint is a faithful copy."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class DeskState(TypedDict, total=False):
    # inputs
    run_id: str
    ticker: str
    debate_rounds: int
    language: str
    profile: dict
    # collect_data
    status: str                      # ok | no_data
    company: dict
    as_of: str
    price: float
    benchmark_price: float | None
    data_source: str
    facts: list[dict]
    warnings: Annotated[list[str], operator.add]
    # recall_memory
    memory: dict
    use_web: bool                    # run web research this time (UI toggle; default from settings)
    research: dict                   # web research: notes (verified sources), sources, dropped, debates
    # analysts (fan-out; merged by key)
    analyst_reports: Annotated[dict[str, dict], merge_dicts]
    # debate
    round: int
    debate: Annotated[list[dict], operator.add]
    # pm + grounding
    decision: dict
    grounding: dict
    revisions: int
    consistency: list[str]
    # finalize
    report: dict[str, Any]
