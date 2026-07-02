"""Model Router — OSIRIS is model-agnostic; this is the seam that makes it so.

Tries providers in priority order and fails over on any error (rate limit,
outage, bad key). Looks like a single BaseLLM to everything upstream, so the
agent/memory/planner never know or care which brain answered.

    OSIRIS ──► ModelRouter ──► gemini ─x─► groq ─x─► ollama ─x─► openai
"""
from __future__ import annotations

import time

from .providers import BaseLLM, LLMResponse, build_chain

RETRY_WAIT = 8          # seconds between full-chain passes (limits are per-minute)
COOLDOWN_RATELIMIT = 90  # skip a rate-limited provider for this long
COOLDOWN_OTHER = 30      # skip a provider with other errors briefly


def _is_ratelimit(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower()


def _reason(e: Exception) -> str:
    """One short human phrase instead of a wall of API JSON."""
    msg = str(e)
    if _is_ratelimit(e):
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
        self._cooldown: dict[str, float] = {}  # provider name -> unix time it may retry
        if not self.chain:
            raise RuntimeError(
                "no LLM providers available — set GEMINI_API_KEY (free: aistudio.google.com/apikey), "
                "GROQ_API_KEY (free: console.groq.com), start Ollama, or set OPENAI_API_KEY.")
        # embeddings come from the first provider that supports them (gemini)
        self.embedder = next((p for p in self.chain if p.name == "gemini"), None)

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        """Defence in depth: skip cooling-down providers, fail over on error,
        and if the whole chain fails, breathe and retry once before giving up."""
        last_error: Exception | None = None
        for attempt in (1, 2):
            for provider in self.chain:
                if attempt == 1 and self._cooldown.get(provider.name, 0) > time.time():
                    continue  # recently rate-limited; skip silently, no error spam
                try:
                    resp = provider.chat(messages, system=system, tools=tools)
                    self.last_provider = provider
                    self._cooldown.pop(provider.name, None)
                    return resp
                except Exception as e:
                    last_error = e
                    cool = COOLDOWN_RATELIMIT if _is_ratelimit(e) else COOLDOWN_OTHER
                    self._cooldown[provider.name] = time.time() + cool
                    self._log(f"[{provider.name}: {_reason(e)} -> resting it {cool}s]")
            if attempt == 1:
                self._log(f"[every provider is busy -- retrying in {RETRY_WAIT}s]")
                time.sleep(RETRY_WAIT)
        raise RuntimeError(
            "all providers are rate-limited or unreachable right now — wait a minute "
            f"and try again (last problem: {_reason(last_error)})") from last_error

    def _log(self, line: str):
        if self.verbose:
            try:  # console encodings vary on Windows; a log line must never kill the request
                print(f"  {line}")
            except Exception:
                pass

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
