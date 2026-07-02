"""Model Router — OSIRIS is model-agnostic; this is the seam that makes it so.

Tries providers in priority order and fails over on any error (rate limit,
outage, bad key). Looks like a single BaseLLM to everything upstream, so the
agent/memory/planner never know or care which brain answered.

    OSIRIS ──► ModelRouter ──► gemini ─x─► groq ─x─► ollama ─x─► openai
"""
from __future__ import annotations

from .providers import BaseLLM, LLMResponse, build_chain


def _reason(e: Exception) -> str:
    """One short human phrase instead of a wall of API JSON."""
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower():
        return "rate/quota limit hit"
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        return "timed out"
    if "401" in msg or "403" in msg or "API key" in msg:
        return "auth problem (check the key in .env)"
    return f"{type(e).__name__}: {msg[:120]}"


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
                resp = provider.chat(messages, system=system, tools=tools)
                self.last_provider = provider
                return resp
            except Exception as e:
                last_error = e
                if self.verbose:
                    try:  # console encodings vary on Windows; a log line must never kill the request
                        print(f"  [{provider.name}: {_reason(e)} -> trying next provider]")
                    except Exception:
                        pass
        raise RuntimeError(f"all providers failed; last error: {last_error}") from last_error

    @property
    def active(self) -> str:
        """Human-readable 'who answered last', e.g. 'groq · llama-3.3-70b-versatile'."""
        p = getattr(self, "last_provider", None)
        if not p:
            return ""
        model = getattr(p, "model", "")
        return f"{p.name} · {model}" if model else p.name

    def embed(self, text: str) -> list[float]:
        if not self.embedder:
            raise NotImplementedError("no embedding-capable provider in chain")
        return self.embedder.embed(text)

    def describe(self) -> str:
        return " → ".join(p.name for p in self.chain)
