"""Provider factory — builds whichever adapters are configured/reachable."""
from __future__ import annotations

import requests

from .. import config
from .base import BaseLLM, LLMResponse, ToolCall  # re-export for convenience

__all__ = ["BaseLLM", "LLMResponse", "ToolCall", "build_chain"]


def _make(name: str) -> BaseLLM | None:
    if name == "gemini" and config.GEMINI_API_KEY:
        from .gemini import GeminiLLM
        return GeminiLLM()
    if name == "groq" and config.GROQ_API_KEY:
        from .openai_compat import OpenAICompatLLM
        return OpenAICompatLLM("groq", "https://api.groq.com/openai/v1",
                               config.GROQ_API_KEY, config.GROQ_MODEL)
    if name == "ollama":
        model = _ollama_model()
        if model:
            from .openai_compat import OpenAICompatLLM
            return OpenAICompatLLM("ollama", f"{config.OLLAMA_HOST}/v1", "ollama", model)
    if name == "openai" and config.OPENAI_API_KEY:
        from .openai_compat import OpenAICompatLLM
        return OpenAICompatLLM("openai", "https://api.openai.com/v1",
                               config.OPENAI_API_KEY, config.OPENAI_MODEL)
    return None


def _ollama_model() -> str | None:
    """Configured model if pulled locally, else the first available; None if down."""
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=1).json()
        names = [m["name"] for m in tags.get("models", [])]
    except (requests.RequestException, ValueError):
        return None
    if not names:
        return None
    return next((n for n in names if n.startswith(config.OLLAMA_MODEL)), names[0])


def build_chain() -> list[BaseLLM]:
    """Instantiate providers from OSIRIS_PROVIDERS priority order, skipping
    any that aren't configured (no key) or reachable (ollama down)."""
    chain = []
    for name in [n.strip().lower() for n in config.PROVIDER_CHAIN.split(",") if n.strip()]:
        p = _make(name)
        if p:
            chain.append(p)
    return chain
