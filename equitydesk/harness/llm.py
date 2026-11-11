"""The metered model client.

Every model call in EquityDesk goes through LLM.structured() or LLM.chat(). Each call:
  1. is refused before it starts if the run's token or cost ceiling is reached (BudgetExceeded);
  2. is written, numbered, to the run's prompt log before it is sent;
  3. is retried with exponential backoff on transient errors (429, 5xx, timeouts, connection);
  4. for structured output, is repaired once if the model's answer fails schema validation;
  5. is priced from pricing.yaml and recorded as one `llm_call` line in the run's trace.

Backends: LangChainBackend (OpenAI, any OpenAI-compatible endpoint, Anthropic) and ScriptedBackend
(offline, deterministic, free; for tests, CI and the UI's demo mode).
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from equitydesk.config import ModelCfg, Settings, plausible_key
from equitydesk.harness.pricing import cost_usd
from equitydesk.harness.trace import Tracer

TRANSIENT_NAMES = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
                   "ServiceUnavailableError", "OverloadedError", "Timeout", "ConnectTimeout", "ReadTimeout",
                   "ConnectError", "RemoteProtocolError"}


class BudgetExceeded(RuntimeError):
    """The run reached its token or cost ceiling. The run stops; its checkpoint can be resumed."""


class ModelError(RuntimeError):
    """A model call failed after all retries, or never produced valid structured output."""


class ConfigError(RuntimeError):
    """The model is not usable as configured (for example, its API key is not set)."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""


@dataclass
class Meter:
    """Running totals for one run, shared by every node (thread-safe: analysts run in parallel)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    unpriced: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, u: Usage, cost: float | None) -> None:
        with self._lock:
            self.input_tokens += u.input_tokens
            self.output_tokens += u.output_tokens
            self.calls += 1
            if cost is None:
                self.unpriced.add(u.model)
            else:
                self.cost_usd += cost

    def snapshot(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "tokens": self.tokens, "cost_usd": round(self.cost_usd, 5), "unpriced": sorted(self.unpriced)}


class Backend(Protocol):
    name: str

    def structured(self, schema: type[BaseModel], messages: list[BaseMessage]) -> tuple[BaseModel | None, str | None, Usage]: ...

    def chat(self, messages: list[BaseMessage]) -> tuple[str, Usage]: ...


# -- real models through LangChain -------------------------------------------------------------

def _usage(msg: Any, fallback_model: str) -> Usage:
    um = getattr(msg, "usage_metadata", None) or {}
    details = um.get("input_token_details") or {}
    meta = getattr(msg, "response_metadata", None) or {}
    return Usage(input_tokens=int(um.get("input_tokens") or 0), output_tokens=int(um.get("output_tokens") or 0),
                 cache_read_tokens=int(details.get("cache_read") or 0),
                 model=meta.get("model_name") or meta.get("model") or fallback_model)


class LangChainBackend:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self.name = cfg.name
        env = cfg.api_key_env or ("ANTHROPIC_API_KEY" if cfg.provider == "anthropic" else "OPENAI_API_KEY")
        key = os.environ.get(env)
        if not plausible_key(key):
            state = "is not set" if not key else f"is set but holds {len(key.strip())} characters, not a key"
            raise ConfigError(f"{env} {state}. Put the key in .env (see .env.example), or switch to the offline "
                              "demo model (provider: scripted).")
        common: dict[str, Any] = {"model": cfg.name, "max_tokens": cfg.max_tokens, "timeout": cfg.timeout_s,
                                  "max_retries": 0, "api_key": key}
        if cfg.temperature is not None:
            common["temperature"] = cfg.temperature
        if cfg.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.model = ChatAnthropic(**common)
        else:
            from langchain_openai import ChatOpenAI
            if cfg.base_url:
                common["base_url"] = cfg.base_url
            self.model = ChatOpenAI(**common)

    def structured(self, schema, messages):
        kwargs = {"include_raw": True}
        if self.cfg.provider != "anthropic":
            kwargs["method"] = "function_calling"   # works across OpenAI-compatible vendors
        out = self.model.with_structured_output(schema, **kwargs).invoke(messages)
        raw, parsed, err = out.get("raw"), out.get("parsed"), out.get("parsing_error")
        usage = _usage(raw, self.cfg.name)
        if parsed is None and err is None:
            err = "the model returned no structured output (no tool call)"
        if err is not None and not isinstance(err, str):
            err = f"{type(err).__name__}: {err}"[:1500]
        return parsed, err, usage

    def chat(self, messages):
        msg = self.model.invoke(messages)
        text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content if isinstance(b, dict))
        return text, _usage(msg, self.cfg.name)


# -- the offline scripted model ----------------------------------------------------------------

FACT_LINE = re.compile(r"^\[([A-Z]\.[A-Za-z0-9_]+)\] ([^:\n]+): (.+?)(?: \(as of [^)]*\))?$", re.M)


class ScriptedBackend:
    """Deterministic answers built from the evidence in the prompt. Free, offline, schema-valid.

    Failure injection for tests: fail_schema_once (first structured call returns invalid output),
    bad_citation_once (first Decision cites an id that does not exist), transient_errors=N (first N
    calls raise a retryable error).
    """

    name = "scripted"

    def __init__(self, fail_schema_once: bool = False, bad_citation_once: bool = False, transient_errors: int = 0,
                 rating: str = "HOLD"):
        self.fail_schema_once, self.bad_citation_once = fail_schema_once, bad_citation_once
        self.transient_errors, self.rating = transient_errors, rating
        self._lock = threading.Lock()

    def _maybe_transient(self):
        with self._lock:
            if self.transient_errors > 0:
                self.transient_errors -= 1
                err = RuntimeError("scripted transient failure")
                err.status_code = 503  # type: ignore[attr-defined]
                raise err

    @staticmethod
    def _facts(messages: list[BaseMessage]) -> list[tuple[str, str, str]]:
        text = "\n".join(m.content for m in messages if isinstance(m.content, str))
        return [(i, l, v) for i, l, v in FACT_LINE.findall(text) if not i.startswith("N.")]

    @staticmethod
    def _usage(messages, out: str) -> Usage:
        n_in = sum(len(m.content) for m in messages if isinstance(m.content, str)) // 4
        return Usage(input_tokens=n_in, output_tokens=len(out) // 4, model="scripted")

    def _points(self, facts, k: int, offset: int = 0) -> list[dict]:
        chosen = facts[offset:offset + k] or facts[:k]
        return [{"claim": f"{label} is {value} [{fid}].", "evidence": [fid]} for fid, label, value in chosen]

    def structured(self, schema, messages):
        from equitydesk.schemas import AnalystReport, DebateTurn, Decision, QAAnswer
        self._maybe_transient()
        with self._lock:
            if self.fail_schema_once:
                self.fail_schema_once = False
                return None, "ValidationError: field 'points' required", self._usage(messages, "{}")
        facts = self._facts(messages)
        sign = 0.0
        for fid, _, v in facts:
            if fid in ("T.ret_3m", "T.rel_strength_3m"):
                sign = 0.3 if not v.startswith("-") else -0.3
                break
        system = messages[0].content.lower() if messages and isinstance(messages[0].content, str) else ""
        if schema is AnalystReport:
            obj = AnalystReport(stance="bullish" if sign > 0 else "bearish" if sign < 0 else "neutral", score=sign,
                                confidence=60, summary=self._points(facts, 1)[0]["claim"] if facts else "No evidence.",
                                points=self._points(facts, 3), data_gaps=[] if facts else ["no evidence supplied"])
        elif schema is DebateTurn:
            bull = "bull" in system.split("\n", 1)[0]
            pts = self._points(facts, 2, offset=0 if bull else 2)
            obj = DebateTurn(argument=("The bull case: " if bull else "The bear case: ") + " ".join(p["claim"] for p in pts),
                             points=pts, rebuttal="The other side overstates its evidence.")
        elif schema is Decision:
            drivers, risks = self._points(facts, 2, 0), self._points(facts, 2, 2)
            with self._lock:
                bad, self.bad_citation_once = self.bad_citation_once, False
            if bad:
                drivers = drivers + [{"claim": "A made-up figure of $999.99 [X.bogus].", "evidence": ["X.bogus"]}]
            obj = Decision(rating=self.rating, conviction=55, horizon_months=6,
                           thesis=(drivers[0]["claim"] if drivers else "Insufficient evidence."),
                           drivers=drivers, risks=risks, catalysts=["Next earnings report"],
                           invalidation=["A close below the protective stop", "A close above 60-day resistance"],
                           memory_note="No previous calls on this ticker and no user notes.")
        elif schema is QAAnswer:
            pts = self._points(facts, 1)
            obj = QAAnswer(answer=pts[0]["claim"] if pts else "The evidence does not cover that.",
                           evidence=[pts[0]["evidence"][0]] if pts else [])
        else:  # pragma: no cover
            raise ModelError(f"scripted backend has no answer for schema {schema.__name__}")
        return obj, None, self._usage(messages, obj.model_dump_json())

    def chat(self, messages):
        self._maybe_transient()
        facts = self._facts(messages)
        q = messages[-1].content if messages else ""
        text = (f"(offline demo model) On \"{q[:80]}\": " +
                (f"{facts[0][1]} is {facts[0][2]} [{facts[0][0]}]." if facts else "the evidence does not cover that."))
        return text, self._usage(messages, text)


# -- the client --------------------------------------------------------------------------------

FATAL_NAMES = {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}


def _status(e: BaseException) -> int | None:
    return getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)


def is_transient(e: BaseException) -> bool:
    if type(e).__name__ in TRANSIENT_NAMES:
        return True
    code = _status(e)
    return code in (408, 409, 429) or (isinstance(code, int) and code >= 500)


def is_fatal(e: BaseException) -> bool:
    """Bad key, no access to the model, unknown model: every further call would fail the same way."""
    return type(e).__name__ in FATAL_NAMES or _status(e) in (401, 403, 404)


def _as_records(messages: list[BaseMessage]) -> list[dict]:
    role = {SystemMessage: "system", HumanMessage: "user", AIMessage: "assistant"}
    return [{"role": role.get(type(m), m.type), "content": m.content} for m in messages]


BackendFactory = Callable[[ModelCfg], Backend]


def default_backend_factory(cfg: ModelCfg) -> Backend:
    return ScriptedBackend() if cfg.provider == "scripted" else LangChainBackend(cfg)


class LLM:
    def __init__(self, settings: Settings, tracer: Tracer, *, meter: Meter | None = None,
                 backend_factory: BackendFactory | None = None, token_ceiling: int | None = None,
                 cost_ceiling: float | None = None, sleep: Callable[[float], None] = time.sleep):
        self.s = settings
        self.tracer = tracer
        self.meter = meter or Meter()
        self.factory = backend_factory or default_backend_factory
        self.token_ceiling = token_ceiling or settings.budget.run_token_ceiling
        self.cost_ceiling = cost_ceiling or settings.budget.run_cost_ceiling_usd
        self._backends: dict[str, Backend] = {}
        self._lock = threading.Lock()
        self._sleep = sleep

    def backend(self, role: str) -> tuple[Backend, ModelCfg]:
        cfg = self.s.model_for(role)
        key = cfg.model_dump_json()
        with self._lock:
            if key not in self._backends:
                self._backends[key] = self.factory(cfg)
            return self._backends[key], cfg

    def _check_budget(self, node: str) -> None:
        m = self.meter
        if m.tokens >= self.token_ceiling or m.cost_usd >= self.cost_ceiling:
            self.tracer.event("budget_exceeded", node=node, tokens=m.tokens, cost_usd=round(m.cost_usd, 5),
                              token_ceiling=self.token_ceiling, cost_ceiling=self.cost_ceiling)
            raise BudgetExceeded(f"run budget reached before {node}: {m.tokens:,} tokens / ${m.cost_usd:.4f} "
                                 f"(ceilings {self.token_ceiling:,} tokens / ${self.cost_ceiling:.2f})")

    def _call(self, node: str, role: str, fn: Callable[[Backend, list[BaseMessage]], Any],
              messages: list[BaseMessage], repair: bool = False) -> Any:
        backend, cfg = self.backend(role)
        attempt = 0
        while True:
            self._check_budget(node)
            attempt += 1
            req = self.tracer.prompt(node, cfg.name, _as_records(messages))
            t0 = time.perf_counter()
            try:
                result, usage = fn(backend, messages)
            except Exception as e:
                ms = round((time.perf_counter() - t0) * 1000)
                transient = is_transient(e)
                self.tracer.event("llm_call", node=node, role=role, model=cfg.name, request=req, attempt=attempt,
                                  ok=False, ms=ms, transient=transient, error=f"{type(e).__name__}: {e}"[:500])
                if transient and attempt <= cfg.max_retries:
                    self._sleep(min(2 ** (attempt - 1) + random.random(), 30))
                    continue
                if is_fatal(e):
                    raise ConfigError(f"{cfg.provider} rejected the request for model {cfg.name} "
                                      f"({type(e).__name__}): check the API key and model name. {e}"[:600]) from e
                raise ModelError(f"{node}: model call failed after {attempt} attempt(s): {type(e).__name__}: {e}") from e
            cost = cost_usd(usage.model or cfg.name, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens)
            self.meter.add(usage, cost)
            ok = not (isinstance(result, tuple) and result[1])
            self.tracer.event("llm_call", node=node, role=role, model=usage.model or cfg.name, request=req,
                              attempt=attempt, ok=ok, repair=repair, ms=round((time.perf_counter() - t0) * 1000),
                              input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                              cache_read_tokens=usage.cache_read_tokens,
                              cost_usd=None if cost is None else round(cost, 6), priced=cost is not None,
                              **({} if ok else {"schema_error": result[1][:500]}))
            return result

    def structured(self, node: str, role: str, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]

        def run(b: Backend, msgs: list[BaseMessage]):
            parsed, err, usage = b.structured(schema, msgs)
            return (parsed, err), usage

        parsed, err = self._call(node, role, run, messages)
        if parsed is None:
            # One repair attempt: tell the model exactly what failed validation.
            messages = messages + [HumanMessage(
                f"Your previous answer failed validation against the {schema.__name__} schema:\n{err}\n"
                f"Answer again with every required field, calling the {schema.__name__} tool exactly once.")]
            parsed, err = self._call(node, role, run, messages, repair=True)
            if parsed is None:
                raise ModelError(f"{node}: no valid {schema.__name__} after a repair attempt: {err}")
        return parsed

    def chat(self, node: str, role: str, system: str, history: list[BaseMessage]) -> str:
        messages = [SystemMessage(system), *history]
        return self._call(node, role, lambda b, m: b.chat(m), messages)

    @staticmethod
    def dump(obj: BaseModel) -> dict:
        return json.loads(obj.model_dump_json())
