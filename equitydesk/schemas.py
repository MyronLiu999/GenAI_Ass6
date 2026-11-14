"""Structured outputs. Every agent answers in one of these schemas; nothing is parsed from free text."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Rating = Literal["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
RATING_SCORE = {"STRONG_SELL": -2, "SELL": -1, "HOLD": 0, "BUY": 1, "STRONG_BUY": 2}
RATING_LABEL = {"STRONG_SELL": "Strong Sell", "SELL": "Sell", "HOLD": "Hold", "BUY": "Buy", "STRONG_BUY": "Strong Buy"}


class Point(BaseModel):
    claim: str = Field(description="One specific claim, citing evidence ids inline like [T.rsi14].")
    evidence: list[str] = Field(description="The evidence ids this claim rests on, e.g. ['T.rsi14', 'T.sma50'].")


class AnalystReport(BaseModel):
    """One analyst's view of the stock from their own slice of the evidence."""

    stance: Literal["bullish", "neutral", "bearish"]
    score: float = Field(ge=-1, le=1, description="-1 strongly bearish .. +1 strongly bullish")
    confidence: int = Field(ge=0, le=100, description="Quality and agreement of the evidence, 0-100")
    summary: str = Field(description="Two or three sentences, with citations.")
    points: list[Point] = Field(description="Two to six key points, each cited.")
    data_gaps: list[str] = Field(default_factory=list, description="Important evidence that is missing.")


class HeadlineView(BaseModel):
    id: str = Field(description="The headline's evidence id, e.g. 'N.3'.")
    relevant: bool = Field(description="False if the headline is really about another company or is generic noise.")
    tone: Literal["positive", "neutral", "negative"] = Field(description="For this stock's investors.")
    materiality: Literal["high", "medium", "low"] = Field(description="Could it change earnings, risk or valuation?")
    why: str = Field(description="One short sentence.")


class SentimentReport(AnalystReport):
    """The sentiment analyst's report, plus a triage of every headline (shown on the News tab)."""

    headlines: list[HeadlineView] = Field(description="One entry for every N.* headline in the evidence.")


class StreetReport(AnalystReport):
    """What the sell-side and independent research say, distilled, and what it implies."""

    consensus_view: str = Field(description="The street's view in three to four sentences: direction, how it is "
                                            "moving (actions, target cuts/raises, estimate revisions), and how fresh it is.")
    street_bull: list[Point] = Field(description="The strongest bull arguments made by researchers, each attributed "
                                                 "to its firm and cited (W/A ids).")
    street_bear: list[Point] = Field(description="The strongest bear arguments made by researchers, attributed and cited.")
    recent_changes: list[Point] = Field(description="What changed in the last 30 days (rating or target moves, "
                                                    "estimate revisions), cited.")
    key_debates: list[str] = Field(description="The two or three questions the street is divided on.")


class EarningsReport(AnalystReport):
    """The earnings and demand analyst: the bar for the next report, the demand behind it, earnings quality."""

    setup_call: Literal["beat likely", "in line", "miss risk", "no data"] = Field(description=(
        "Your call on the next report against the consensus, from the bar (E.q_bar), the beat record, guidance "
        "and the demand drivers."))
    setup_view: str = Field(description=(
        "Two or three sentences: what the consensus asks for against what the company has been delivering and "
        "guiding, what the demand data implies, and so what you expect, with citations."))
    demand_drivers: list[Point] = Field(default_factory=list, description=(
        "The quantified demand and supply drivers (customer spending, market size, guidance, capacity), each "
        "with its figure and what it implies for this company's revenue, cited (D/E/Q ids)."))
    quality_flags: list[Point] = Field(default_factory=list, description=(
        "Earnings-quality issues the headline numbers hide (receivables, inventory, cash conversion, "
        "non-operating gains), cited Q ids; empty if none."))


class DriverItem(BaseModel):
    source_ids: list[str] = Field(description="The [S<n>] markers this figure comes from. Required.")
    source: str = Field(description="Who published or stated the figure (company, firm, outlet).")
    date: str = Field(description="YYYY-MM-DD if known, else the best available date.")
    kind: Literal["company guidance", "customer spending", "market size", "supply and capacity", "pricing",
                  "competition", "expectations", "regulation", "other"]
    metric: str = Field(description="What is measured, e.g. 'Combined 2026 capex guidance of the four largest "
                                    "US cloud providers'.")
    figure: str = Field(description=("The number as the source states it, with currency and unit, abbreviated and "
                                     "with no words around it, e.g. '$400B', '$1.3T', '+30%'."))
    period: str = Field(default="", description="The period it refers to, e.g. 'calendar 2026'.")
    context: str = Field(default="", description="One sentence: why the source says it matters for the company.")


class DriverNotes(BaseModel):
    items: list[DriverItem]


class ResearchNote(BaseModel):
    source_ids: list[str] = Field(description="The [S<n>] markers this note comes from. Required; notes without "
                                              "one are discarded.")
    source: str = Field(description="Firm, author or publication, e.g. 'Baird' or 'Morningstar'.")
    date: str = Field(description="YYYY-MM-DD if known, else the best available date.")
    kind: Literal["sell-side note", "rating change", "independent research", "news analysis"]
    stance: Literal["bullish", "neutral", "bearish"]
    rating: str = Field(default="", description="The rating as stated (e.g. 'Neutral (from Outperform)').")
    price_target: float | None = Field(default=None, description="New price target, if stated.")
    prior_target: float | None = Field(default=None, description="Previous price target, if stated.")
    thesis: str = Field(description="The core argument in one or two sentences.")
    key_points: list[str] = Field(default_factory=list, description="Up to four supporting arguments.")


class ResearchNotes(BaseModel):
    notes: list[ResearchNote]
    debates: list[str] = Field(default_factory=list, description="Debates the sources describe, with [S<n>] markers.")


class NewsTriage(BaseModel):
    items: list[HeadlineView]


SCHEMAS = {"AnalystReport": AnalystReport, "SentimentReport": SentimentReport, "StreetReport": StreetReport,
           "EarningsReport": EarningsReport}


class DebateTurn(BaseModel):
    """One side's argument in the bull/bear debate."""

    argument: str = Field(description="The case in one paragraph, with citations.")
    points: list[Point] = Field(description="Two to four strongest points, each cited.")
    rebuttal: str = Field(default="", description="Direct answer to the other side's last argument, if any.")


class Decision(BaseModel):
    """The portfolio manager's final call."""

    rating: Rating
    conviction: int = Field(ge=0, le=100)
    horizon_months: int = Field(ge=1, le=24)
    thesis: str = Field(description="The investment thesis in three to five sentences, with citations.")
    drivers: list[Point] = Field(description=(
        "Two to five reasons for the rating, ranked. Each is an inference, not a restatement: it connects at least "
        "two facts from different categories (for example the consensus bar [E] against the run-rate [Q], or "
        "customer spending [D] against next-year estimates [E]) and says what that means for the stock."))
    risks: list[Point] = Field(description="Two to five risks to the thesis, each cited.")
    catalysts: list[str] = Field(default_factory=list, description="Upcoming events that could move the stock.")
    invalidation: list[str] = Field(min_length=2, description=(
        "At least two concrete conditions that would make this call wrong: at least one on the upside and one on the "
        "downside, each with a price level taken from the evidence (stop, support, resistance, a moving average)."))
    memory_note: str = Field(min_length=10, description=(
        "Required. How your previous calls on this ticker (hits or misses) and the user's notes shaped this call; "
        "if there is no history and no note, say so explicitly."))
    override_rationale: str = Field(default="", description="Required when the rating departs from the analysts' consensus.")
    key_debates: list[str] = Field(min_length=1, description=(
        "The two or three questions the outcome hinges on, each with the side you take and why, cited."))
    expectations_gap: str = Field(default="", description=(
        "What the street expects for the next report and next year against what the run-rate, the beat record, "
        "guidance and the demand drivers imply (cite E/Q/D ids); your call (beat, in line or miss) and what "
        "the stock does if you are right."))
    variant_view: str = Field(min_length=10, description=(
        "Where this call differs from the street (cite A/W ids) and why the street may be wrong; or, if you agree "
        "with the street, what you see that it does not price in."))


class QAAnswer(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)


# -- market monitor --------------------------------------------------------------------------------

MonitorCategory = Literal["macro", "policy", "geopolitics", "company", "industry", "market", "other"]


class MonitorScore(BaseModel):
    id: str = Field(description="The item's id, e.g. 'M3'.")
    score: int = Field(ge=1, le=10, description="Importance for an investor holding the watchlist (rubric in the prompt).")
    category: MonitorCategory
    tickers: list[str] = Field(default_factory=list, description="Watchlist tickers it could move; only from the list.")
    title: str = Field(description="The headline rewritten in the target language: faithful, short, no added claims.")
    rumor: bool = Field(description="True if the item itself reports unconfirmed information (reportedly, sources "
                                    "say, people familiar, rumour, 据传, 知情人士).")
    why: str = Field(default="", description="One short phrase: why this score.")


class MonitorScores(BaseModel):
    items: list[MonitorScore]


class OverviewPoint(BaseModel):
    text: str = Field(description="One bullet, in the target language, stating what happened and what it means.")
    ids: list[str] = Field(description="The item ids (M...) this bullet rests on. Required.")


class TickerPoints(BaseModel):
    ticker: str
    points: list[OverviewPoint]


class MonitorOverview(BaseModel):
    macro: list[OverviewPoint] = Field(description="Three to six bullets on macro, policy and geopolitics.")
    tickers: list[TickerPoints] = Field(default_factory=list, description="For watchlist stocks with news only.")
    watch_next: list[OverviewPoint] = Field(default_factory=list, description="Up to three things to watch next.")
