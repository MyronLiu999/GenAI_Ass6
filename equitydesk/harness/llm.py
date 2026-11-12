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
from equitydesk.harness.pricing import cost_usd, tool_price
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
    search_calls: int = 0          # hosted web-search tool calls, priced per call


class Unsupported(RuntimeError):
    """The configured model or provider cannot do this (e.g. hosted web search). The caller degrades."""


@dataclass
class Meter:
    """Running totals for one run, shared by every node (thread-safe: analysts run in parallel)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    searches: int = 0
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
            self.searches += u.search_calls
            if cost is None:
                self.unpriced.add(u.model)
            else:
                self.cost_usd += cost

    def snapshot(self) -> dict:
        return {"calls": self.calls, "searches": self.searches, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "tokens": self.tokens, "cost_usd": round(self.cost_usd, 5),
                "unpriced": sorted(self.unpriced)}


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
        self._key = key
        common: dict[str, Any] = {"model": cfg.name, "max_tokens": cfg.max_tokens, "timeout": cfg.timeout_s,
                                  "max_retries": 0, "api_key": key}
        if cfg.temperature is not None and not cfg.reasoning_effort:   # reasoning models reject temperature
            common["temperature"] = cfg.temperature
        if cfg.reasoning_effort and cfg.provider == "openai":
            # reasoning models take function tools only through the Responses API
            common["use_responses_api"] = True
            common["reasoning"] = {"effort": cfg.reasoning_effort}
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

    def web_search(self, system: str, prompt: str, context_size: str = "medium") -> tuple[dict, Usage]:
        """OpenAI's hosted web-search tool (Responses API). Returns the answer text with each cited span
        replaced by a source marker [S<n>], and the sources the tool actually cited. Structured output is
        deliberately not requested here: combined with search it returns no citations and invents URLs."""
        if self.cfg.provider != "openai" or self.cfg.base_url:
            raise Unsupported(f"hosted web search needs provider openai on api.openai.com (configured: "
                              f"{self.cfg.provider}{' via ' + self.cfg.base_url if self.cfg.base_url else ''})")
        import openai

        from equitydesk.policy.provenance import mark_citations
        client = openai.OpenAI(api_key=self._key, max_retries=0, timeout=self.cfg.timeout_s * 2)
        extra = {"reasoning": {"effort": self.cfg.reasoning_effort}} if self.cfg.reasoning_effort else {}
        r = client.responses.create(model=self.cfg.name, instructions=system, input=prompt,
                                    tools=[{"type": "web_search", "search_context_size": context_size}],
                                    max_output_tokens=max(self.cfg.max_tokens, 2500), **extra)
        text, annotations = "", []
        for item in r.output:
            if item.type != "message":
                continue
            for part in item.content:
                if getattr(part, "type", "") != "output_text":
                    continue
                offset = len(text)
                text += part.text
                annotations += [{"url": a.url, "title": getattr(a, "title", "") or "", "start": a.start_index + offset,
                                 "end": a.end_index + offset}
                                for a in (part.annotations or []) if getattr(a, "type", "") == "url_citation"]
        marked, sources = mark_citations(text, annotations)
        details = getattr(r.usage, "input_tokens_details", None)
        usage = Usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                      cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0), model=r.model,
                      search_calls=sum(1 for o in r.output if o.type == "web_search_call"))
        return {"text": marked, "sources": sources}, usage


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
                 rating: str = "HOLD", shallow_once: bool = False):
        self.fail_schema_once, self.bad_citation_once = fail_schema_once, bad_citation_once
        self.shallow_once = shallow_once      # first Decision restates single figures (fails the depth check)
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
        return [(i, l, v) for i, l, v in FACT_LINE.findall(text) if i[0] not in ("N", "W", "D")]

    @staticmethod
    def _usage(messages, out: str) -> Usage:
        n_in = sum(len(m.content) for m in messages if isinstance(m.content, str)) // 4
        return Usage(input_tokens=n_in, output_tokens=len(out) // 4, model="scripted")

    def _points(self, facts, k: int, offset: int = 0) -> list[dict]:
        chosen = facts[offset:offset + k] or facts[:k]
        return [{"claim": f"{label} is {value} [{fid}].", "evidence": [fid]} for fid, label, value in chosen]

    @staticmethod
    def _linked(facts, k: int) -> list[dict]:
        """Points that connect two facts from different categories (what the depth check asks of the PM)."""
        out, used = [], set()
        for i, (a, la, va) in enumerate(facts):
            if a in used:
                continue
            b = next(((fb, lb, vb) for fb, lb, vb in facts[i + 1:] if fb[0] != a[0] and fb not in used), None)
            if b is None:
                continue
            used |= {a, b[0]}
            out.append({"claim": f"{la} is {va} [{a}] while {b[1]} is {b[2]} [{b[0]}].", "evidence": [a, b[0]]})
            if len(out) == k:
                break
        return out

    def structured(self, schema, messages):
        from equitydesk.schemas import (AnalystReport, DebateTurn, Decision, DriverItem, DriverNotes,
                                        EarningsReport, HeadlineView, QAAnswer, ResearchNote, ResearchNotes,
                                        SentimentReport, StreetReport)
        self._maybe_transient()
        with self._lock:
            if self.fail_schema_once:
                self.fail_schema_once = False
                return None, "ValidationError: field 'points' required", self._usage(messages, "{}")
        facts = self._facts(messages)
        text = "\n".join(m.content for m in messages if isinstance(m.content, str))
        sign = 0.0
        for fid, _, v in facts:
            if fid in ("T.ret_3m", "T.rel_strength_3m"):
                sign = 0.3 if not v.startswith("-") else -0.3
                break
        system = messages[0].content.lower() if messages and isinstance(messages[0].content, str) else ""
        if issubclass(schema, AnalystReport):
            base = dict(stance="bullish" if sign > 0 else "bearish" if sign < 0 else "neutral", score=sign,
                        confidence=60, summary=self._points(facts, 1)[0]["claim"] if facts else "No evidence.",
                        points=self._points(facts, 3), data_gaps=[] if facts else ["no evidence supplied"])
            if schema is SentimentReport:
                ids = list(dict.fromkeys(re.findall(r"\[(N\.\d+)\]", text)))
                base["headlines"] = [HeadlineView(id=i, relevant=True, tone="neutral", materiality="low",
                                                  why="Offline demo triage.") for i in ids]
            elif schema is StreetReport:
                base.update(consensus_view="Offline demo: the street view is summarised from the A.* evidence.",
                            street_bull=self._points(facts, 1), street_bear=self._points(facts, 1, 1),
                            recent_changes=self._points(facts, 1, 2), key_debates=["Offline demo debate."])
            elif schema is EarningsReport:
                bar = next((v for f, _, v in facts if f == "E.q_bar"), None)
                base.update(setup_call="in line" if bar else "no data",
                            setup_view=(f"Offline demo: {bar} [E.q_bar]." if bar else "Offline demo: no estimates."),
                            demand_drivers=self._points(facts, 1, 1), quality_flags=[])
            obj = schema(**base)
        elif schema.__name__ == "NewsTriage":
            ids = list(dict.fromkeys(re.findall(r"\[(H\d+)\]", text)))
            obj = schema(items=[HeadlineView(id=i, relevant=True, tone="neutral", materiality="low",
                                             why="Offline demo triage.") for i in ids])
        elif schema is ResearchNotes:
            ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", text)))
            obj = ResearchNotes(notes=[ResearchNote(source_ids=[i], source=f"Demo Research {i}", date="2026-11-01",
                                                    kind="sell-side note", stance="neutral", rating="Hold",
                                                    thesis="Offline demo note.", key_points=["Demo point."])
                                       for i in ids], debates=["Offline demo debate."])
        elif schema is DriverNotes:
            ids = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", text)))
            obj = DriverNotes(items=[DriverItem(source_ids=[i], source=f"Demo Industry {i}", date="2026-11-01",
                                                kind="customer spending", metric="Demo customer capex guidance",
                                                figure="$100 billion", period="2026",
                                                context="Offline demo driver.") for i in ids])
        elif schema is DebateTurn:
            bull = "bull" in system.split("\n", 1)[0]
            pts = self._points(facts, 2, offset=0 if bull else 2)
            obj = DebateTurn(argument=("The bull case: " if bull else "The bear case: ") + " ".join(p["claim"] for p in pts),
                             points=pts, rebuttal="The other side overstates its evidence.")
        elif schema is Decision:
            drivers, risks = (self._linked(facts, 2) or self._points(facts, 2, 0)), self._points(facts, 2, 2)
            with self._lock:
                bad, self.bad_citation_once = self.bad_citation_once, False
            if bad:
                drivers = drivers + [{"claim": "A made-up figure of $999.99 [X.bogus].", "evidence": ["X.bogus"]}]
            with self._lock:
                shallow, self.shallow_once = self.shallow_once, False
            if shallow:
                drivers = self._points(facts, 2, 0)
            obj = Decision(rating=self.rating, conviction=55, horizon_months=6,
                           thesis=(drivers[0]["claim"] if drivers else "Insufficient evidence."),
                           drivers=drivers, risks=risks, catalysts=["Next earnings report"],
                           invalidation=["A close below the protective stop", "A close above 60-day resistance"],
                           memory_note="No previous calls on this ticker and no user notes.",
                           key_debates=["Offline demo: does growth justify the multiple?"],
                           variant_view="Offline demo: in line with the street consensus.",
                           expectations_gap=("Offline demo: " + next((f"{v} [E.q_bar]." for f, _, v in facts
                                                                      if f == "E.q_bar"), "no consensus estimates.")))
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

    def web_search(self, system: str, prompt: str, context_size: str = "medium") -> tuple[dict, Usage]:
        self._maybe_transient()
        text = ("Demo Securities reiterated a Hold rating on 2026-11-01 [S1]. "
                "Sample Research argued the valuation already reflects the growth outlook [S2].")
        sources = [{"id": "S1", "url": "https://example.com/demo-note-1", "title": "Offline demo note 1",
                    "domain": "example.com"},
                   {"id": "S2", "url": "https://example.com/demo-note-2", "title": "Offline demo note 2",
                    "domain": "example.com"}]
        return {"text": text, "sources": sources}, Usage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4,
                                                         model="scripted", search_calls=1)


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

    def backend(self, role: str, node: str | None = None) -> tuple[Backend, ModelCfg]:
        cfg = self.s.model_for(role, node)
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
        backend, cfg = self.backend(role, node)
        attempt = 0
        while True:
            self._check_budget(node)
            attempt += 1
            req = self.tracer.prompt(node, cfg.name, _as_records(messages))
            t0 = time.perf_counter()
            try:
                result, usage = fn(backend, messages)
            except Unsupported:
                raise
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
            if usage.search_calls and usage.model != "scripted":      # the offline model's searches are free
                per_call = tool_price("web_search")
                cost = None if cost is None or per_call is None else cost + usage.search_calls * per_call
            self.meter.add(usage, cost)
            ok = not (isinstance(result, tuple) and result[1])
            self.tracer.event("llm_call", node=node, role=role, model=usage.model or cfg.name, request=req,
                              attempt=attempt, ok=ok, repair=repair, ms=round((time.perf_counter() - t0) * 1000),
                              input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                              cache_read_tokens=usage.cache_read_tokens, search_calls=usage.search_calls,
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

    def web_search(self, node: str, role: str, system: str, prompt: str, context_size: str = "medium") -> dict:
        """Hosted web search, metered like any call. Raises Unsupported when the provider cannot search."""
        backend, _ = self.backend(role, node)
        if not hasattr(backend, "web_search"):
            raise Unsupported(f"backend {backend.name} has no web search")
        messages = [SystemMessage(system), HumanMessage(prompt)]
        return self._call(node, role, lambda b, m: b.web_search(system, prompt, context_size), messages)

    @staticmethod
    def dump(obj: BaseModel) -> dict:
        return json.loads(obj.model_dump_json())
