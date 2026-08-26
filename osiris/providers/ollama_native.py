"""Ollama native adapter — talks to /api/chat directly instead of the
shared OpenAI-compatible adapter every other provider uses, specifically
to unlock think:false.

Why this exists, not just another provider reusing OpenAICompatLLM: Ollama's
own OpenAI-compatible endpoint (what every other provider — Groq, Cerebras,
Mistral, OpenRouter, GitHub Models, OpenAI, and Ollama itself until this
file — calls through) silently ignores think:false, verified live: still
~78s per call on a reasoning model with it set. The parameter IS honored on
Ollama's native /api/chat, verified live: 100s -> 10s on the same model,
same task, same correct answer. That gap is real and measured (see
benchmarks/FINDINGS.md in this repo's history), and only closes by calling
the native API directly for Ollama specifically — hence a separate small
adapter, not an Ollama special-case bolted onto the shared one.

Message/tool-call format is close to OpenAI's — verified live: native
/api/chat accepts the same messages/tools shape and round-trips assistant
tool_calls + tool results correctly — but with two real wire-format
differences, not cosmetic ones, both handled below:
  - the response wraps the answer in "message" directly, not "choices[0].message"
  - tool_call arguments arrive as a dict, not a JSON-encoded string
"""
from __future__ import annotations

import json

import requests

from .. import config
from .base import BaseLLM, LLMResponse, ToolCall
from .openai_compat import _rescue_json_call, _rescue_pseudo_calls

# Same rationale as openai_compat.py: a 400 here is usually the model
# flubbing tool-call syntax once — retryable. Ollama's own error text for
# this hasn't been observed to differ from what the OpenAI-compat path
# reports (same underlying llama.cpp tool-call grammar either way), so the
# same match strings apply; harmless even if a given message doesn't match —
# it just skips the retry and raises normally.
_RETRYABLE_400 = ("tool call validation failed", "failed_generation", "Failed to call a function")


def _to_native_messages(messages: list[dict], system: str) -> list[dict]:
    out = [{"role": "system", "content": system}] if system else []
    counter = 0        # deterministic ids for calls that came from id-less providers (Gemini)
    pending_ids: list[str] = []  # queue consumed by the tool results that follow
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                calls = []
                pending_ids = []
                for tc in m["tool_calls"]:
                    tid = tc.id or f"call_{counter}"
                    counter += 1
                    pending_ids.append(tid)
                    # dict, not json.dumps(tc.args) — native /api/chat wants
                    # arguments as a real object on the wire, verified live.
                    calls.append({"id": tid, "type": "function",
                                  "function": {"name": tc.name, "arguments": tc.args}})
                msg["tool_calls"] = calls
            out.append(msg)
        elif m["role"] == "tool":
            tid = m.get("tool_call_id") or (pending_ids.pop(0) if pending_ids else "call_0")
            out.append({"role": "tool", "tool_call_id": tid, "content": m["content"]})
    return out


class OllamaNativeLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str, host: str = "", think: bool | None = None):
        self.model = model
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.think = config.OLLAMA_THINK if think is None else think

    def chat(self, messages, system="", tools=None) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": _to_native_messages(messages, system),
            "think": self.think,
            "stream": False,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        for attempt in (1, 2):
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
            if r.status_code == 400 and attempt == 1 and any(s in r.text for s in _RETRYABLE_400):
                continue  # model garbled a tool call; generation is stochastic — one more try
            if r.status_code >= 400:  # surface the API's actual complaint, not just the code
                raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
            break
        msg = r.json()["message"]

        out = LLMResponse(text=msg.get("content") or "")
        for tc in msg.get("tool_calls") or []:
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):  # defensive: not observed live, but don't crash if it happens
                try:
                    args = json.loads(args) or {}
                except json.JSONDecodeError:
                    args = {}
            out.tool_calls.append(ToolCall(name=tc["function"]["name"], args=args, id=tc.get("id", "")))
        if "<function=" in out.text:
            out = _rescue_pseudo_calls(out)
        if not out.tool_calls and out.text.lstrip().startswith("{") and '"name"' in out.text:
            out = _rescue_json_call(out)
        return out

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(f"{self.name} adapter does not provide embeddings")
