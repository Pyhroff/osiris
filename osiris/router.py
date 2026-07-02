"""Model Router — OSIRIS is model-agnostic; this is the seam that makes it so.

Tries providers in priority order and fails over on any error (rate limit,
outage, bad key). Looks like a single BaseLLM to everything upstream, so the
agent/memory/planner never know or care which brain answered.

    OSIRIS ──► ModelRouter ──► gemini ─x─► groq ─x─► ollama ─x─► openai
"""
from __future__ import annotations

from .providers import BaseLLM, LLMResponse, build_chain


class ModelRouter(BaseLLM):
    name = "router"

    def __init__(self, chain: list[BaseLLM] | None = None, verbose: bool = True):
        self.chain = chain if chain is not None else build_chain()
        self.verbose = verbose
        if not self.chain:
            raise RuntimeError(
                "no LLM providers available — set GEMINI_API_KEY (free: aistudio.google.com/apikey), "
                "GROQ_API_KEY (free: console.groq.com), start Ollama, or set OPENAI_API_KEY.")
        # embeddings come from the first provider that supports them (gemini)
        self.embedder = next((p for p in self.chain if p.name == "gemini"), None)

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        last_error: Exception | None = None
        for provider in self.chain:
            try:
                return provider.chat(messages, system=system, tools=tools)
            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"  [{provider.name} failed ({type(e).__name__}), failing over]")
        raise RuntimeError(f"all providers failed; last error: {last_error}") from last_error

    def embed(self, text: str) -> list[float]:
        if not self.embedder:
            raise NotImplementedError("no embedding-capable provider in chain")
        return self.embedder.embed(text)

    def describe(self) -> str:
        return " → ".join(p.name for p in self.chain)
