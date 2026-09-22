"""
Provider-selection AND provider-fallback tests for llm.py. No real API keys
and no network calls -- constructing a ChatGroq/ChatOpenAI/ChatGoogleGenerativeAI
client only stores config locally, it doesn't call out to the provider, so
selection tests are safe with fake key strings. Fallback tests use small fake
client objects instead, to control exactly when a provider "fails" without
depending on any real provider's actual current error behavior.

Run: pytest tests/test_llm_provider.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pydantic import BaseModel
import llm


@pytest.fixture(autouse=True)
def _reset_llm_cache_and_env(monkeypatch):
    """get_llm() caches its result in module-level globals -- clear them before
    each test so provider selection actually re-runs, and clear all three
    provider env vars first so tests control exactly what's "configured"."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    llm._llm = None
    llm._providers = None
    yield
    llm._llm = None
    llm._providers = None


# ---------------------------------------------------------------------------
# Provider selection (which providers get built, in what order)
# ---------------------------------------------------------------------------

def test_no_key_raises_clear_error():
    with pytest.raises(RuntimeError, match="No LLM API key found"):
        llm.get_llm()


def test_groq_only_selected(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake_groq_key")
    providers = llm._build_provider_list()
    assert [p[0] for p in providers] == ["groq"]


def test_nvidia_only_selected(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake_nvidia_key")
    providers = llm._build_provider_list()
    assert [p[0] for p in providers] == ["nvidia"]
    assert providers[0][1].openai_api_base == "https://integrate.api.nvidia.com/v1"


def test_google_only_selected(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_google_key")
    providers = llm._build_provider_list()
    assert [p[0] for p in providers] == ["google"]


def test_priority_order_when_all_three_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake1")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake2")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake3")
    providers = llm._build_provider_list()
    assert [p[0] for p in providers] == ["groq", "nvidia", "google"]


def test_get_llm_is_cached_across_calls(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake_groq_key")
    first = llm.get_llm()
    second = llm.get_llm()
    assert first is second


def test_get_llm_wraps_all_configured_providers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake1")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake3")
    client = llm.get_llm()
    assert isinstance(client, llm._FallbackChatModel)
    assert [p[0] for p in client._providers] == ["groq", "google"]


# ---------------------------------------------------------------------------
# Fallback behavior -- fake clients, no real provider involved
# ---------------------------------------------------------------------------

class _FakeClient:
    """Stand-in for a langchain chat model. Raises on invoke if configured to
    fail; otherwise returns a marker string so tests can tell which provider
    actually answered."""

    def __init__(self, name, fail=False, fail_structured=False):
        self.name = name
        self.fail = fail
        self.fail_structured = fail_structured
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return f"response-from-{self.name}"

    def with_structured_output(self, schema):
        return _FakeStructuredClient(self, schema)


class _FakeStructuredClient:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, messages):
        self.parent.calls += 1
        if self.parent.fail_structured:
            raise RuntimeError(f"{self.parent.name} structured output failed")
        return f"structured-from-{self.parent.name}"


class _DummySchema(BaseModel):
    value: str = "x"


def test_primary_succeeds_no_fallback_needed():
    """No unnecessary provider calls: if the first provider works, the second
    is never touched."""
    primary = _FakeClient("primary")
    backup = _FakeClient("backup")
    wrapper = llm._FallbackChatModel([("primary", primary), ("backup", backup)])
    assert wrapper.invoke(["hi"]) == "response-from-primary"
    assert primary.calls == 1
    assert backup.calls == 0


def test_primary_fails_falls_back_to_secondary():
    primary = _FakeClient("primary", fail=True)
    backup = _FakeClient("backup")
    wrapper = llm._FallbackChatModel([("primary", primary), ("backup", backup)])
    assert wrapper.invoke(["hi"]) == "response-from-backup"
    assert primary.calls == 1  # tried exactly once, not retried
    assert backup.calls == 1


def test_first_two_fail_falls_back_to_third_google():
    groq = _FakeClient("groq", fail=True)
    nvidia = _FakeClient("nvidia", fail=True)
    google = _FakeClient("google")
    wrapper = llm._FallbackChatModel([("groq", groq), ("nvidia", nvidia), ("google", google)])
    assert wrapper.invoke(["hi"]) == "response-from-google"
    assert groq.calls == 1
    assert nvidia.calls == 1
    assert google.calls == 1


def test_all_providers_fail_raises_and_does_not_fabricate():
    a = _FakeClient("a", fail=True)
    b = _FakeClient("b", fail=True)
    wrapper = llm._FallbackChatModel([("a", a), ("b", b)])
    with pytest.raises(RuntimeError, match="b failed"):
        wrapper.invoke(["hi"])
    assert a.calls == 1
    assert b.calls == 1


def test_structured_output_falls_back_on_failure():
    """Intent extraction never silently degrades to free-form parsing -- a
    provider that can't fulfil the structured call is skipped, the next
    provider's structured call is used instead."""
    primary = _FakeClient("primary", fail_structured=True)
    backup = _FakeClient("backup")
    wrapper = llm._FallbackChatModel([("primary", primary), ("backup", backup)])
    result = wrapper.with_structured_output(_DummySchema).invoke(["hi"])
    assert result == "structured-from-backup"


def test_structured_output_no_unnecessary_call_when_primary_succeeds():
    primary = _FakeClient("primary")
    backup = _FakeClient("backup")
    wrapper = llm._FallbackChatModel([("primary", primary), ("backup", backup)])
    result = wrapper.with_structured_output(_DummySchema).invoke(["hi"])
    assert result == "structured-from-primary"
    assert backup.calls == 0
