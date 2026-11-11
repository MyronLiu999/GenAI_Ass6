import pytest

from equitydesk.harness.llm import LLM, BudgetExceeded, ModelError, ScriptedBackend, is_transient
from equitydesk.harness.pricing import cost_usd, price_line
from equitydesk.harness.trace import Tracer, read_prompts, read_trace
from equitydesk.schemas import AnalystReport

USER = "<evidence>\n[T.price] Last close: $100.00 (as of 2026-11-10)\n[T.ret_3m] 3-month return: 5.0%\n</evidence>"


def make(settings, tmp_path, backend, **kw):
    tracer = Tracer("t-run", tmp_path / "runs")
    return LLM(settings, tracer, backend_factory=lambda cfg: backend, sleep=lambda s: None, **kw), tracer


def test_pricing_longest_prefix():
    assert price_line("gpt-4o-mini-2024-07-18")["input"] == 0.15
    assert price_line("gpt-4o-2024-08-06")["input"] == 2.50
    assert price_line("no-such-model") is None
    assert cost_usd("gpt-4o", 1_000_000, 100_000) == pytest.approx(2.5 + 1.0)
    assert cost_usd("gpt-4o", 1_000_000, 0, cache_read_tokens=1_000_000) == pytest.approx(1.25)


def test_structured_call_is_traced_and_metered(settings, tmp_path):
    llm, tracer = make(settings, tmp_path, ScriptedBackend())
    rep = llm.structured("technical", "analyst", AnalystReport, "system", USER)
    assert isinstance(rep, AnalystReport) and rep.points[0].evidence == ["T.price"]
    calls = [e for e in read_trace("t-run", tracer.dir) if e["type"] == "llm_call"]
    assert len(calls) == 1 and calls[0]["ok"] and calls[0]["request"] == 1 and calls[0]["priced"]
    assert read_prompts("t-run", tracer.dir)[0]["messages"][1]["content"] == USER
    assert llm.meter.calls == 1 and llm.meter.tokens > 0


def test_transient_errors_are_retried(settings, tmp_path):
    llm, tracer = make(settings, tmp_path, ScriptedBackend(transient_errors=2))
    llm.structured("technical", "analyst", AnalystReport, "system", USER)
    calls = [e for e in read_trace("t-run", tracer.dir) if e["type"] == "llm_call"]
    assert [c["ok"] for c in calls] == [False, False, True] and calls[0]["transient"]


def test_retries_are_bounded(settings, tmp_path):
    llm, _ = make(settings, tmp_path, ScriptedBackend(transient_errors=10))   # max_retries=2 in the test settings
    with pytest.raises(ModelError, match="after 3 attempt"):
        llm.structured("technical", "analyst", AnalystReport, "system", USER)


def test_schema_failure_is_repaired_once(settings, tmp_path):
    llm, tracer = make(settings, tmp_path, ScriptedBackend(fail_schema_once=True))
    llm.structured("technical", "analyst", AnalystReport, "system", USER)
    calls = [e for e in read_trace("t-run", tracer.dir) if e["type"] == "llm_call"]
    assert len(calls) == 2 and calls[0].get("schema_error") and calls[1]["repair"]
    assert "failed validation" in read_prompts("t-run", tracer.dir)[1]["messages"][-1]["content"]


def test_budget_ceiling_refuses_the_next_call(settings, tmp_path):
    llm, tracer = make(settings, tmp_path, ScriptedBackend(), token_ceiling=10)
    llm.structured("technical", "analyst", AnalystReport, "system", USER)   # first call is allowed
    with pytest.raises(BudgetExceeded):
        llm.structured("technical", "analyst", AnalystReport, "system", USER)
    assert any(e["type"] == "budget_exceeded" for e in read_trace("t-run", tracer.dir))


def test_is_transient():
    e = RuntimeError("x")
    e.status_code = 429
    assert is_transient(e)
    e.status_code = 400
    assert not is_transient(e)


def test_missing_api_key_is_a_config_error(settings, monkeypatch):
    from equitydesk.config import ModelCfg
    from equitydesk.harness.llm import ConfigError, LangChainBackend
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        LangChainBackend(ModelCfg(provider="openai", name="gpt-4o", api_key_env="OPENAI_API_KEY"))


def test_placeholder_api_key_is_rejected(monkeypatch):
    from equitydesk.config import ModelCfg
    from equitydesk.harness.llm import ConfigError, LangChainBackend
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    with pytest.raises(ConfigError, match="holds 0 characters"):
        LangChainBackend(ModelCfg(provider="openai", name="gpt-4o", api_key_env="OPENAI_API_KEY"))


def test_auth_errors_stop_the_run_instead_of_degrading(settings, tmp_path):
    from equitydesk.harness.llm import ConfigError

    class Unauthorized(ScriptedBackend):
        def structured(self, schema, messages):
            e = RuntimeError("invalid api key")
            e.status_code = 401
            raise e

    llm, _ = make(settings, tmp_path, Unauthorized())
    with pytest.raises(ConfigError, match="check the API key"):
        llm.structured("technical", "analyst", AnalystReport, "system", USER)


def test_env_file_overrides_placeholder_but_not_real_values(tmp_path, monkeypatch):
    import equitydesk.config as config
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-real-key-0123456789abcdef\nOTHER=from-file\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OTHER", "from-env")
    config.load_env()
    import os
    assert os.environ["OPENAI_API_KEY"].startswith("sk-real") and os.environ["OTHER"] == "from-env"


def test_role_overrides(settings):
    s = settings.model_copy(update={"roles": {"analyst": {"name": "cheap"}}})
    assert s.model_for("analyst").name == "cheap" and s.model_for("pm").name == "scripted"
    assert s.clamp_rounds(99) == s.graph.max_debate_rounds and s.clamp_rounds(0) == 1
