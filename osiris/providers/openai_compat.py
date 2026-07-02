"""One adapter for every OpenAI-compatible chat API: Groq, Ollama, OpenAI.

They all speak POST {base_url}/chat/completions with the same tool-calling
schema, so three providers cost one implementation.
"""
from __future__ import annotations

import json

import requests

from .base import BaseLLM, LLMResponse, ToolCall


def _to_openai_messages(messages: list[dict], system: str) -> list[dict]:
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
                    calls.append({"id": tid, "type": "function",
                                  "function": {"name": tc.name, "arguments": json.dumps(tc.args)}})
                msg["tool_calls"] = calls
            out.append(msg)
        elif m["role"] == "tool":
            tid = m.get("tool_call_id") or (pending_ids.pop(0) if pending_ids else "call_0")
            out.append({"role": "tool", "tool_call_id": tid, "content": m["content"]})
    return out


class OpenAICompatLLM(BaseLLM):
    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, messages, system="", tools=None):
        payload: dict = {
            "model": self.model,
            "messages": _to_openai_messages(messages, system),
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload, timeout=180)  # generous: local models load cold
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]

        out = LLMResponse(text=msg.get("content") or "")
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            out.tool_calls.append(ToolCall(name=tc["function"]["name"], args=args, id=tc.get("id", "")))
        return out

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(f"{self.name} adapter does not provide embeddings")
