"""The grounding check: is every claim cited, does every citation exist, does every number match?

Run on every analyst report (warnings) and on the PM decision (blocking: a failed decision is sent
back to the PM with the problems listed, up to `graph.max_revisions` times).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from equitydesk.config import GroundingCfg
from equitydesk.tools.evidence import Fact

CITE_RE = re.compile(r"\[\s*([A-Z]\.[A-Za-z0-9_]+(?:\s*[,;]\s*[A-Z]\.[A-Za-z0-9_]+)*)\s*\]")  # [A] or [A, B]
ID_RE = re.compile(r"[A-Z]\.[A-Za-z0-9_]+")
# Figures resting on sell-side estimates; when S.consensus_check flags them as stale they may not drive a call.
ESTIMATE_IDS = {"F.pe_fwd", "F.peg", "S.target_mean", "S.target_upside", "S.analyst_rec"}


def cite_ids(text: str) -> list[str]:
    return [i for g in CITE_RE.findall(text or "") for i in ID_RE.findall(g)]


def normalize_text(text: str) -> str:
    """'[A, B]' -> '[A][B]': one citation format everywhere downstream."""
    return CITE_RE.sub(lambda m: "".join(f"[{i}]" for i in ID_RE.findall(m.group(1))), text or "")


def normalize_point(pt: dict) -> dict:
    """Canonical citations, and every id in `evidence` also shown inline (models often leave them out)."""
    claim = normalize_text((pt.get("claim") or "").strip())
    inline = cite_ids(claim)
    evidence = list(dict.fromkeys(list(pt.get("evidence") or []) + inline))
    missing = [i for i in evidence if i not in inline]
    if missing:
        claim = f"{claim} " + "".join(f"[{i}]" for i in missing)
    return {**pt, "claim": claim, "evidence": evidence}
NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([-+−])?(\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|x\b|×|B\b|bn\b|billion\b)?")


@dataclass
class GroundingReport:
    passed: bool = True
    points_checked: int = 0
    points_cited: int = 0
    unknown_ids: list[str] = field(default_factory=list)
    uncited_points: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    numbers_checked: int = 0
    stale_drivers: list[str] = field(default_factory=list)

    @property
    def citation_coverage(self) -> float:
        return 1.0 if not self.points_checked else self.points_cited / self.points_checked

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation_coverage"] = round(self.citation_coverage, 3)
        return d

    def feedback(self) -> str:
        lines = []
        if self.unknown_ids:
            lines.append(f"- These cited ids do not exist in the evidence pack: {', '.join(sorted(set(self.unknown_ids)))}. "
                         "Cite only ids that appear in <evidence> or <untrusted>.")
        if self.uncited_points:
            lines.append("- These points cite no evidence; add a valid citation or remove them: "
                         + "; ".join(f'"{p[:80]}"' for p in self.uncited_points))
        if self.ungrounded_numbers:
            lines.append("- These numbers do not match any evidence value; use the figures exactly as given "
                         "or drop them: " + "; ".join(self.ungrounded_numbers[:8]))
        if self.stale_drivers:
            lines.append("- S.consensus_check flags the sell-side targets and forward estimates as probably stale, yet "
                         "these drivers rest on them: " + "; ".join(f'"{p[:80]}"' for p in self.stale_drivers)
                         + ". Move them to risks or caveats, or replace them with drivers based on reported figures.")
        return "\n".join(lines)


PERIOD_AFTER = re.compile(r"\s?-?\s?(day|week|month|year|quarter|session|trading day|d|w|m|y|q)s?\b", re.I)
DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
# "1-day 95% VaR", "at a 99% confidence level": the percentage names the statistic, it is not a claim
LEVEL_PCT = re.compile(r"\s*(?:(?:one|1)-day |daily |historical )*(?:VaR|value.at.risk|confidence|CI)\b", re.I)


def _is_label_integer(text: str, start: int, end: int, v: float) -> bool:
    """A bare integer that names a period or a parameter rather than claims a figure:
    small counts, years, '200-day', '12m', '6 months', 'RSI(14)', 'MACD (12,26,9)'."""
    if abs(v) < 10 or 1900 <= v <= 2100:
        return True
    after, before = text[end:end + 16], text[max(0, start - 1):start]
    if after[:1].isalpha() or PERIOD_AFTER.match(after):
        return True
    return before in ("(", ",") or after[:1] in (")", ",")


def _numbers(text: str) -> list[tuple[str, float, bool]]:
    """(raw, value, strict) for each number worth checking. strict = has a decimal, %, $ or multiple suffix;
    those are always checked. Bare integers are checked unless they are labels (see _is_label_integer)."""
    text = DATE_RE.sub(" ", CITE_RE.sub(" ", text))
    out = []
    for m in NUM_RE.finditer(text):
        sign, dollar, whole, dec, suffix = m.groups()
        if suffix == "%" and LEVEL_PCT.match(text, m.end()):
            continue
        try:
            v = float(whole.replace(",", "") + (dec or ""))
        except ValueError:
            continue
        if sign in ("-", "−"):
            v = -v
        strict = bool(dollar or dec or suffix)
        if not strict and _is_label_integer(text, m.start(), m.end(), v):
            continue
        out.append((m.group(0).strip(), v, strict))
    return out


def _matches(v: float, candidates: list[float], rel: float, integer_claim: bool) -> bool:
    for c in candidates:
        tol = max(abs(c) * rel, 0.05)
        if integer_claim:
            tol = max(tol, 0.5)
        if abs(abs(v) - abs(c)) <= tol:
            return True
    return False


def check_points(points: list[dict], facts: dict[str, Fact], cfg: GroundingCfg, extra_text: str = "") -> GroundingReport:
    rep = GroundingReport()
    values = [f.numeric for f in facts.values() if f.numeric is not None]
    texts = []
    for pt in points:
        claim = (pt.get("claim") or "").strip()
        ids = set(pt.get("evidence") or []) | set(cite_ids(claim))
        rep.points_checked += 1
        known = [i for i in ids if i in facts]
        rep.unknown_ids += [i for i in ids if i not in facts]
        if known:
            rep.points_cited += 1
        else:
            rep.uncited_points.append(claim)
        texts.append(claim)
    if extra_text:
        rep.unknown_ids += [i for i in cite_ids(extra_text) if i not in facts]
        texts.append(extra_text)
    for t in texts:
        for raw, v, strict in _numbers(t):
            rep.numbers_checked += 1
            if not _matches(v, values, cfg.number_rel_tolerance, integer_claim="." not in raw):
                rep.ungrounded_numbers.append(raw)
    rep.passed = not rep.unknown_ids and not rep.uncited_points
    return rep


def check_analyst(report: dict, facts: dict[str, Fact], cfg: GroundingCfg) -> GroundingReport:
    rep = check_points(report.get("points") or [], facts, cfg, extra_text=report.get("summary", ""))
    rep.passed = rep.passed and len(rep.ungrounded_numbers) <= cfg.max_ungrounded_numbers
    return rep


def check_decision(decision: dict, facts: dict[str, Fact], cfg: GroundingCfg) -> GroundingReport:
    drivers = list(decision.get("drivers") or [])
    points = drivers + list(decision.get("risks") or [])
    extra = " ".join([decision.get("thesis", ""), decision.get("override_rationale", ""),
                      *(decision.get("invalidation") or []), *(decision.get("catalysts") or [])])
    rep = check_points(points, facts, cfg, extra_text=extra)
    if "S.consensus_check" in facts:
        rep.stale_drivers = [d.get("claim", "") for d in drivers
                             if (set(d.get("evidence") or []) | set(cite_ids(d.get("claim", "")))) & ESTIMATE_IDS]
    rep.passed = (rep.passed and len(rep.ungrounded_numbers) <= cfg.max_ungrounded_numbers
                  and not rep.stale_drivers)
    return rep


def consistency_warnings(decision: dict, analyst_reports: dict[str, dict]) -> list[str]:
    """Flags a PM call that contradicts the desk without saying why."""
    from equitydesk.schemas import RATING_SCORE

    scores = [r["score"] for r in analyst_reports.values() if not r.get("failed")]
    if not scores or not decision:
        return []
    mean = sum(scores) / len(scores)
    pm = RATING_SCORE.get(decision.get("rating", "HOLD"), 0) / 2
    warn = []
    if abs(pm - mean) >= 0.75 and not (decision.get("override_rationale") or "").strip():
        warn.append(f"PM rating {decision.get('rating')} departs from the analysts' mean score {mean:+.2f} "
                    "without an override rationale.")
    if len(scores) >= 3 and (max(scores) - min(scores)) >= 1.2:
        warn.append("Analysts strongly disagree (score spread >= 1.2); treat conviction with care.")
    return warn
