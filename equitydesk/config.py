"""Settings: settings.yaml validated into typed models. The one place configuration is read."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ROLES = ("analyst", "debate", "pm", "qa", "research", "monitor")


class ModelCfg(BaseModel):
    provider: Literal["openai", "anthropic", "scripted", "claude_code"] = "openai"
    name: str = "gpt-4o"
    temperature: float | None = 0.2     # null for models that reject it (reasoning models)
    max_tokens: int = 2500
    timeout_s: float = 90
    max_retries: int = Field(3, ge=0, le=8)
    api_key_env: str | None = None
    base_url: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None   # reasoning models (gpt-5.x, o3)


class BudgetCfg(BaseModel):
    run_token_ceiling: int = 150_000
    run_cost_ceiling_usd: float = 1.50
    qa_token_ceiling: int = 60_000


class GraphCfg(BaseModel):
    debate_rounds: int = Field(2, ge=1)
    max_debate_rounds: int = Field(4, ge=1)
    max_revisions: int = Field(1, ge=0, le=3)
    language: Literal["English", "Chinese"] = "English"


class GroundingCfg(BaseModel):
    max_ungrounded_numbers: int = 2
    number_rel_tolerance: float = 0.03
    hold_band_pct: float = 5.0
    min_days_to_score: int = 5      # a younger call is "too early" in memory and the track record


class DataCfg(BaseModel):
    provider: Literal["yfinance", "fixtures"] = "yfinance"
    fixtures_dir: str = "fixtures"
    history_period: str = "2y"
    benchmark: str = "SPY"
    cache_ttl_minutes: int = 30
    news_limit: int = 10
    staleness_days: int = 5
    risk_free_rate_pct: float = 4.0
    atr_stop_multiple: float = 2.0


class ResearchCfg(BaseModel):
    web_search: bool = True                  # hosted web search for research notes (OpenAI only)
    context_size: Literal["low", "medium", "high"] = "medium"   # more context = deeper, more tokens
    max_notes: int = Field(8, ge=1, le=20)
    lookback_days: int = Field(60, ge=7, le=365)
    drivers: bool = True                     # a second search for quantified industry and demand drivers
    max_drivers: int = Field(10, ge=1, le=20)


class NotifyCfg(BaseModel):
    desktop: bool = True                     # Windows toast notifications on this machine
    ntfy_topic: str = ""                     # optional phone push via ntfy.sh: subscribe to this topic in the app


class MonitorCfg(BaseModel):
    """The market monitor (equitydesk/monitor): a background process that collects, scores and pushes news."""
    interval_s: int = Field(60, ge=15, le=3600)
    engine: Literal["openai", "claude_code"] = "openai"   # claude_code: the logged-in Claude Code CLI, no API cost
    claude_models: dict[str, str] = {"score": "haiku", "overview": "sonnet", "report": "sonnet"}
    language: str = "Chinese"                # titles, overview and reports are written in this language
    push_min_score: int = Field(8, ge=1, le=10)
    push_holding_min_score: int = Field(7, ge=1, le=10)   # lower bar for news about a stock you hold
    max_pushes_per_hour: int = Field(8, ge=0, le=100)
    report_min_score: int = Field(8, ge=1, le=10)
    max_reports_per_day: int = Field(6, ge=0, le=50)
    daily_cost_ceiling_usd: float = Field(2.0, ge=0)
    overview_every_min: int = Field(120, ge=10)
    quotes_every_min: int = Field(5, ge=1)
    social_every_min: int = Field(15, ge=5)
    predictions_every_min: int = Field(15, ge=5)
    keep_days: int = Field(14, ge=1)
    sources: dict[str, bool] = {}            # per-source switches, e.g. {wscn: false}; default all on
    polymarket_queries: list[str] = ["Fed decision", "recession", "tariffs"]
    kalshi_series: list[str] = ["KXFEDDECISION"]
    notify: NotifyCfg = NotifyCfg()


class PathsCfg(BaseModel):
    data_dir: str = "data"


class Settings(BaseModel):
    model: ModelCfg = ModelCfg()
    roles: dict[str, dict[str, Any]] = {}
    budget: BudgetCfg = BudgetCfg()
    graph: GraphCfg = GraphCfg()
    grounding: GroundingCfg = GroundingCfg()
    data: DataCfg = DataCfg()
    research: ResearchCfg = ResearchCfg()
    monitor: MonitorCfg = MonitorCfg()
    paths: PathsCfg = PathsCfg()

    # -- derived ---------------------------------------------------------------------------------
    def model_for(self, role: str, node: str | None = None) -> ModelCfg:
        """The model for a role: `model`, with that role's overrides from `roles` merged on top, and then a
        node's own overrides (e.g. `roles: {earnings: {...}}` for just the earnings analyst)."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        merged = self.model.model_dump() | dict(self.roles.get(role) or {})
        if node and node not in ROLES and node in self.roles:
            merged |= dict(self.roles[node])
        return ModelCfg(**merged)

    def _abs(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    @property
    def data_dir(self) -> Path:
        return self._abs(self.paths.data_dir)

    @property
    def fixtures_dir(self) -> Path:
        return self._abs(self.data.fixtures_dir)

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def store_path(self) -> Path:
        return self.data_dir / "desk.db"

    @property
    def monitor_path(self) -> Path:
        return self.data_dir / "monitor.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    def clamp_rounds(self, requested: int | None) -> int:
        """Debate rounds are bounded in code, whatever the UI or caller asks for."""
        n = self.graph.debate_rounds if requested is None else int(requested)
        return max(1, min(n, self.graph.max_debate_rounds))


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: str | Path | None = None, overrides: dict | None = None) -> Settings:
    """settings.yaml (or $EQUITYDESK_SETTINGS), with optional overrides deep-merged on top."""
    path = Path(path or os.environ.get("EQUITYDESK_SETTINGS") or ROOT / "settings.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return Settings(**_deep_merge(raw or {}, overrides or {}))


def plausible_key(value: str | None) -> bool:
    """API keys are long; a blank or few-character value is a leftover placeholder, not a key."""
    return bool(value) and len(value.strip()) >= 20


def load_env() -> None:
    """Load .env from the project root. A real variable already set wins, except a blank or placeholder
    one (a stale system variable must not shadow the key in .env)."""
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover
        return
    for k, v in dotenv_values(ROOT / ".env").items():
        if v is None or v == "":
            continue
        cur = os.environ.get(k)
        if cur is None or (k.endswith("_KEY") and not plausible_key(cur)):
            os.environ[k] = v
