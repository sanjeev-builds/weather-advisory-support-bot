"""
LLM client setup. Provider priority is GROQ -> NVIDIA -> GOOGLE, controlled
entirely by which *_API_KEY variables are set. graph.py never branches on
provider -- every node just calls get_llm() and uses structured output.

Fallback behavior: get_llm() returns a small wrapper holding EVERY configured
provider (not just the first one). On each .invoke() / .with_structured_output()
.invoke() call, it tries providers in priority order and returns the first
success. A provider failing for any reason -- bad/expired key, rate limit,
a deprecated or forbidden model, a transient network error, or a structured-
output call it can't fulfill -- simply moves on to the next configured
provider. Each provider is tried at most once per call (no retry loop); if
every configured provider fails, the last error is raised, which is exactly
what graph.py's existing try/except blocks around every LLM call site
already handle (extract_intent -> honest "intent_error" reply, the composite-
SOP loop -> skip that one SOP, compose_reply -> deterministic fallback text).
None of that error handling needed to change -- this file is the only one
that does.

Why one broad except instead of classifying auth vs. rate-limit vs.
model-not-found errors separately: every one of those cases calls for the
same action here (try the next provider), so classifying them wouldn't
change the behavior, only add code that has to be kept in sync with each
provider's specific exception types.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_llm = None
_providers = None


def _build_provider_list():
    """Ordered list of (name, client) for every provider with a configured
    key. Constructing a client is local/offline for all three providers here
    (no network call happens until .invoke()), so building the whole list up
    front is cheap and lets the fallback wrapper try them without any retry
    loop or repeated construction."""
    providers = []

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        # Model availability varies by Groq account/key -- this was picked by
        # querying https://api.groq.com/openai/v1/models with the configured key
        # rather than assumed, since Groq deprecates/rotates model names over time.
        providers.append(("groq", ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
        )))

    if os.getenv("NVIDIA_API_KEY"):
        from langchain_openai import ChatOpenAI
        # NVIDIA NIM (build.nvidia.com) exposes an OpenAI-compatible endpoint,
        # so this reuses langchain_openai rather than needing a dedicated package.
        # NVIDIA's model catalog changes over time (a prior audit found the
        # previously-configured default deprecated, and several currently-listed
        # models forbidden for this account) -- rather than guess a new model
        # string, NVIDIA now just falls back to the next provider automatically
        # if it fails, same as any other provider failure.
        providers.append(("nvidia", ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
            temperature=0,
        )))

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        providers.append(("google", ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )))

    return providers


class _FallbackChatModel:
    """Tries each configured provider in order for a single call; stops at
    the first success. Never calls the same provider twice for one request,
    never loops, never retries beyond the fixed, short provider list."""

    def __init__(self, providers):
        self._providers = providers

    def _try_each(self, call):
        last_exc = None
        for name, client in self._providers:
            try:
                return call(client)
            except Exception as exc:
                print(f"[llm.py] provider '{name}' failed ({type(exc).__name__}); trying next provider")
                last_exc = exc
        raise last_exc

    def invoke(self, messages):
        return self._try_each(lambda client: client.invoke(messages))

    def with_structured_output(self, schema):
        return _FallbackStructured(self._providers, schema)


class _FallbackStructured:
    """Same fallback behavior as _FallbackChatModel, for the structured-output
    path (.with_structured_output(schema).invoke(...)) that extract_intent and
    the composite-SOP judgment both use. If one provider can't fulfil the
    structured call at all, that's caught the same way as any other failure
    and the next provider is tried -- intent extraction never silently
    degrades to unstructured/free-form parsing."""

    def __init__(self, providers, schema):
        self._providers = providers
        self._schema = schema

    def invoke(self, messages):
        last_exc = None
        for name, client in self._providers:
            try:
                return client.with_structured_output(self._schema).invoke(messages)
            except Exception as exc:
                print(f"[llm.py] provider '{name}' structured-output call failed ({type(exc).__name__}); trying next provider")
                last_exc = exc
        raise last_exc


def get_llm():
    global _llm, _providers
    if _llm is not None:
        return _llm

    _providers = _build_provider_list()
    if not _providers:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (recommended, free at "
            "console.groq.com), NVIDIA_API_KEY (build.nvidia.com), or "
            "GOOGLE_API_KEY in your .env file."
        )

    _llm = _FallbackChatModel(_providers)
    return _llm
