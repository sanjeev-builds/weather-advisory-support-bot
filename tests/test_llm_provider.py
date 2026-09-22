"""
Provider-selection tests for llm.py. No real API keys and no network calls --
constructing a ChatGroq/ChatOpenAI/ChatGoogleGenerativeAI client only stores
config locally, it doesn't call out to the provider, so these are safe and
fast to run with fake key strings.

These do NOT prove a given provider's API actually works (that needs a real
key and a real call, e.g. via eval/eval_suite.py) -- they prove get_llm()
picks the right provider class in the right priority order, which is the one
thing that's easy to get wrong silently (e.g. a typo'd elif branch order).

Run: pytest tests/test_llm_provider.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import llm


@pytest.fixture(autouse=True)
def _reset_llm_cache_and_env(monkeypatch):
    """get_llm() caches its result in a module-level global -- clear it before
    each test so provider selection actually re-runs, and clear all three
    provider env vars first so tests control exactly what's "configured"."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    llm._llm = None
    yield
    llm._llm = None


def test_groq_selected_when_groq_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake_groq_key")
    client = llm.get_llm()
    assert type(client).__name__ == "ChatGroq"


def test_groq_takes_priority_over_nvidia_and_google(monkeypatch):
    """All three keys set at once -- Groq must still win (documented priority)."""
    monkeypatch.setenv("GROQ_API_KEY", "fake_groq_key")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake_nvidia_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_google_key")
    client = llm.get_llm()
    assert type(client).__name__ == "ChatGroq"


def test_nvidia_selected_when_only_nvidia_key_present(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake_nvidia_key")
    client = llm.get_llm()
    assert type(client).__name__ == "ChatOpenAI"
    assert client.openai_api_base == "https://integrate.api.nvidia.com/v1"


def test_nvidia_takes_priority_over_google(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake_nvidia_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_google_key")
    client = llm.get_llm()
    assert type(client).__name__ == "ChatOpenAI"


def test_google_selected_as_last_resort(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_google_key")
    client = llm.get_llm()
    assert type(client).__name__ == "ChatGoogleGenerativeAI"


def test_no_key_raises_clear_error():
    with pytest.raises(RuntimeError, match="No LLM API key found"):
        llm.get_llm()


def test_client_is_cached_across_calls(monkeypatch):
    """get_llm() should not rebuild the client on every call within one process."""
    monkeypatch.setenv("GROQ_API_KEY", "fake_groq_key")
    first = llm.get_llm()
    second = llm.get_llm()
    assert first is second
