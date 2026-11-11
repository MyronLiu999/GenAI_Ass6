"""The final report: the deterministic trade plan and the Markdown rendering (download / archive)."""

from __future__ import annotations

from equitydesk.schemas import RATING_LABEL, RATING_SCORE
from equitydesk.tools.evidence import Fact, fmt, index


def build_plan(decision: dict, idx: dict[str, Fact], profile: dict) -> dict:
    """What to do, in code: size scales with conviction (50% of max size at 0, 100% at 100)."""
    score = RATING_SCORE.get(decision.get("rating", "HOLD"), 0)
    conviction = int(decision.get("conviction") or 0)
    get = lambda k: idx[k].numeric if k in idx else None  # noqa: E731
    price, stop, stop_pct = get("T.price"), get("P.stop"), get("P.stop_pct")
    max_pct, max_shares = get("P.position_pct"), get("P.shares")
    plan = {"action": "", "price": price, "stop": stop, "stop_pct": stop_pct, "size_pct": None, "shares": None,
            "risk_usd": None, "support": get("T.support_60d"), "resistance": get("T.resistance_60d"),
            "rule": "size = max size for risk budget x (0.5 + conviction/200); stop = ATR multiple below price"}
    if score > 0 and max_pct is not None and price:
        factor = 0.5 + conviction / 200
        plan["size_pct"] = round(max_pct * factor, 2)
        plan["shares"] = int((max_shares or 0) * factor)
        plan["risk_usd"] = round(plan["shares"] * (price - (stop or price)), 2)
        plan["action"] = "Open or add to a long position" + (" (full size)" if score == 2 else "")
    elif score == 0:
        plan["action"] = "No new position. Existing holders: hold with the protective stop."
    else:
        plan["action"] = "Reduce or avoid. No new long position" + (" (exit)" if score == -2 else "")
    return plan


def _points(points: list[dict]) -> list[str]:
    return [f"- {p['claim']}" for p in points or []]


def render_markdown(r: dict) -> str:
    company = r.get("company") or {}
    lines = [f"# {company.get('name', r['ticker'])} ({r['ticker']}) — Equity Research Desk",
             f"_Generated {r.get('generated_at')} · data as of {r.get('as_of') or 'n/a'} · source {r.get('data_source')} · "
             f"run `{r['run_id']}`_", ""]
    if r.get("status") != "ok":
        lines += ["**No report: the desk could not get usable market data.**", ""]
        lines += [f"- {w}" for w in r.get("warnings", [])]
        lines += ["", f"> {r['disclaimer']}"]
        return "\n".join(lines)
    d, plan, g = r["decision"], r["plan"], r.get("grounding") or {}
    lines += [f"## Rating: **{RATING_LABEL[d['rating']]}** · conviction {d['conviction']}/100 · horizon "
              f"{d['horizon_months']} months", "", f"Last close {fmt(r.get('price'), '$')}.", "",
              "### Thesis", d["thesis"], "", "### Drivers", *_points(d.get("drivers")), "",
              "### Risks", *_points(d.get("risks")), ""]
    if d.get("catalysts"):
        lines += ["### Catalysts", *[f"- {c}" for c in d["catalysts"]], ""]
    if d.get("invalidation"):
        lines += ["### What would make this call wrong", *[f"- {c}" for c in d["invalidation"]], ""]
    lines += ["### Trade plan (computed in code)", f"- Action: {plan['action']}"]
    if plan.get("stop"):
        lines.append(f"- Protective stop: {fmt(plan['stop'], '$')} ({fmt(plan['stop_pct'], '%')} below price)")
    if plan.get("size_pct") is not None:
        lines.append(f"- Size: {plan['size_pct']:.1f}% of portfolio, {plan['shares']:,} shares, "
                     f"{fmt(plan['risk_usd'], '$')} at risk if stopped")
    if plan.get("support") and plan.get("resistance"):
        lines.append(f"- 60-day range: support {fmt(plan['support'], '$')}, resistance {fmt(plan['resistance'], '$')}")
    lines += ["", f"_Rule: {plan['rule']}._", ""]
    if d.get("memory_note"):
        lines += ["### Memory", d["memory_note"], ""]
    if d.get("override_rationale"):
        lines += ["### Override rationale", d["override_rationale"], ""]
    lines += ["## Analyst views", ""]
    titles = {"fundamental": "Fundamental", "technical": "Technical", "sentiment": "Sentiment & News", "risk": "Risk"}
    for name, rep in (r.get("analysts") or {}).items():
        if rep.get("failed"):
            lines += [f"### {titles.get(name, name)}: failed", rep.get("error", ""), ""]
            continue
        lines += [f"### {titles.get(name, name)}: {rep['stance']} (score {rep['score']:+.2f}, confidence {rep['confidence']})",
                  rep["summary"], *_points(rep.get("points")), ""]
    if r.get("debate"):
        lines += ["## Bull vs bear debate", ""]
        for t in r["debate"]:
            lines += [f"**Round {t['round']} · {t['side'].upper()}:** {t['argument']}", ""]
    lines += ["## Quality checks",
              f"- Grounding: {'PASSED' if g.get('passed') else 'FAILED'} · citation coverage "
              f"{g.get('citation_coverage', 0):.0%} · numbers checked {g.get('numbers_checked', 0)} · unmatched "
              f"{len(g.get('ungrounded_numbers') or [])} · revisions {g.get('revisions', 0)}"]
    lines += [f"- {w}" for w in r.get("consistency", []) + r.get("warnings", [])]
    u = r.get("usage") or {}
    if u:
        lines.append(f"- Model usage: {u.get('calls')} calls, {u.get('tokens', 0):,} tokens, ${u.get('cost_usd', 0):.4f}")
    lines += ["", "## Evidence", "| id | item | value | as of |", "|---|---|---|---|"]
    for f in index(r.get("facts") or []).values():
        label = f.label.replace("|", "/")
        lines.append(f"| {f.id} | {label} | {f.display().replace('|', '/')} | {f.as_of} |")
    lines += ["", f"> {r['disclaimer']}"]
    return "\n".join(lines)
