"""
LLM client setup. Groq (free tier, fast Llama models) is the default because
that's what the .env.example ships with. NVIDIA NIM and Gemini are drop-in
fallbacks, tried in that order, if GROQ_API_KEY isn't set -- useful if Groq's
free-tier rate limit is hit (NVIDIA's free tier has more daily headroom, at
the cost of slower per-request latency). Swapping providers never touches
graph.py -- every node just calls get_llm() and uses structured output.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_llm = None


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        # Model availability varies by Groq account/key -- this was picked by
        # querying https://api.groq.com/openai/v1/models with the configured key
        # rather than assumed, since Groq deprecates/rotates model names over time.
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
        )
    elif os.getenv("NVIDIA_API_KEY"):
        from langchain_openai import ChatOpenAI
        # NVIDIA NIM (build.nvidia.com) exposes an OpenAI-compatible endpoint,
        # so this reuses langchain_openai rather than needing a dedicated package.
        _llm = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
            temperature=0,
        )
    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )
    else:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (recommended, free at "
            "console.groq.com), NVIDIA_API_KEY (build.nvidia.com), or "
            "GOOGLE_API_KEY in your .env file."
        )

    return _llm
