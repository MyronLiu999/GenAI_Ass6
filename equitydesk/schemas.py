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
    drivers: list[Point] = Field(description="Two to five reasons for the rating, each cited.")
    risks: list[Point] = Field(description="Two to five risks to the thesis, each cited.")
    catalysts: list[str] = Field(default_factory=list, description="Upcoming events that could move the stock.")
    invalidation: list[str] = Field(min_length=2, description=(
        "At least two concrete conditions that would make this call wrong: at least one on the upside and one on the "
        "downside, each with a price level taken from the evidence (stop, support, resistance, a moving average)."))
    memory_note: str = Field(min_length=10, description=(
        "Required. How your previous calls on this ticker (hits or misses) and the user's notes shaped this call; "
        "if there is no history and no note, say so explicitly."))
    override_rationale: str = Field(default="", description="Required when the rating departs from the analysts' consensus.")


class QAAnswer(BaseModel):
    answer: str
    evidence: list[str] = Field(default_factory=list)
