"""Provider factory — builds whichever adapters are configured/reachable."""
from __future__ import annotations

import requests

from .. import config
from .base import BaseLLM, LLMResponse, ToolCall  # re-export for convenience

__all__ = ["BaseLLM", "LLMResponse", "ToolCall", "build_chain"]


# OpenAI-compatible endpoints: name -> (base_url, key getter, model getter)
_COMPAT = {
    "groq": ("https://api.groq.com/openai/v1",
             lambda: config.GROQ_API_KEY, lambda: config.GROQ_MODEL),
    "cerebras": ("https://api.cerebras.ai/v1",
                 lambda: config.CEREBRAS_API_KEY, lambda: config.CEREBRAS_MODEL),
    "mistral": ("https://api.mistral.ai/v1",
                lambda: config.MISTRAL_API_KEY, lambda: config.MISTRAL_MODEL),
    "openrouter": ("https://openrouter.ai/api/v1",
                   lambda: config.OPENROUTER_API_KEY, lambda: config.OPENROUTER_MODEL),
    "github": ("https://models.inference.ai.azure.com",
               lambda: config.GITHUB_MODELS_TOKEN, lambda: config.GITHUB_MODELS_MODEL),
    "openai": ("https://api.openai.com/v1",
               lambda: config.OPENAI_API_KEY, lambda: config.OPENAI_MODEL),
}


def _make(name: str) -> BaseLLM | None:
    if name == "gemini" and config.GEMINI_API_KEY:
        from .gemini import GeminiLLM
        return GeminiLLM()
    if name == "ollama":
        model = _ollama_model()
        if model:
            from .openai_compat import OpenAICompatLLM
            return OpenAICompatLLM("ollama", f"{config.OLLAMA_HOST}/v1", "ollama", model)
        return None
    if name in _COMPAT:
        base_url, key, model = _COMPAT[name]
        if key():
            from .openai_compat import OpenAICompatLLM
            return OpenAICompatLLM(name, base_url, key(), model())
    return None


# Local models ranked by tool-calling competence. gemma is absent on purpose:
# it has no tool support, which matters for an agent runtime.
_OLLAMA_RANK = ["qwen3", "qwen2.5", "llama3.3", "llama3.1", "phi4", "mistral", "llama3.2", "llama3"]


def _pick_ollama(names: list[str]) -> str | None:
    if not names:
        return None
    if config.OLLAMA_MODEL:  # explicit user choice wins
        hit = next((n for n in names if n.startswith(config.OLLAMA_MODEL)), None)
        if hit:
            return hit
    for prefix in _OLLAMA_RANK:
        hit = next((n for n in names if n.startswith(prefix)), None)
        if hit:
            return hit
    return names[0]


def _ollama_model() -> str | None:
    """Best tool-capable local model, or None if Ollama is down."""
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=1).json()
        names = [m["name"] for m in tags.get("models", [])]
    except (requests.RequestException, ValueError):
        return None
    return _pick_ollama(names)


def build_chain() -> list[BaseLLM]:
    """Instantiate providers from OSIRIS_PROVIDERS priority order, skipping
    any that aren't configured (no key) or reachable (ollama down)."""
    chain = []
    for name in [n.strip().lower() for n in config.PROVIDER_CHAIN.split(",") if n.strip()]:
        p = _make(name)
        if p:
            chain.append(p)
    return chain
