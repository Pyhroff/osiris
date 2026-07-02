"""One adapter for every OpenAI-compatible chat API: Groq, Ollama, OpenAI.

They all speak POST {base_url}/chat/completions with the same tool-calling
schema, so three providers cost one implementation.
"""
from __future__ import annotations

import json
import re

import requests

from .base import BaseLLM, LLMResponse, ToolCall

# Llama-family models sometimes emit tool calls as text like
#   <function=remember {"text": "..."}</function>
# instead of using the structured tool_calls field. Rescue those.
_PSEUDO_CALL = re.compile(r"<function=(\w+)\s*(\{.*?\})?\s*(?:</function>|/?>)", re.DOTALL)


def _rescue_pseudo_calls(out: LLMResponse) -> LLMResponse:
    for m in _PSEUDO_CALL.finditer(out.text):
        try:
            args = json.loads(m.group(2)) if m.group(2) else {}
        except json.JSONDecodeError:
            continue
        out.tool_calls.append(ToolCall(name=m.group(1), args=args or {}))
    out.text = _PSEUDO_CALL.sub("", out.text).replace("</function>", "").strip()
    return out


def _rescue_json_call(out: LLMResponse) -> LLMResponse:
    """Small local models sometimes reply with a bare JSON tool call as text,
    e.g. {"name": "recall", "parameters": {...}}. Convert it if well-formed."""
    try:
        obj = json.loads(out.text)
    except json.JSONDecodeError:
        return out
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("parameters") or obj.get("arguments") or {}
        if isinstance(args, dict):
            out.tool_calls.append(ToolCall(name=obj["name"], args=args))
            out.text = ""
    return out


# a 400 here is usually the model flubbing tool-call syntax once — retryable
_RETRYABLE_400 = ("tool call validation failed", "failed_generation", "Failed to call a function")


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

        for attempt in (1, 2):
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload, timeout=180)  # generous: local models load cold
            if r.status_code == 400 and attempt == 1 and any(s in r.text for s in _RETRYABLE_400):
                continue  # model garbled a tool call; generation is stochastic — one more try
            if r.status_code >= 400:  # surface the API's actual complaint, not just the code
                raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
            break
        msg = r.json()["choices"][0]["message"]

        out = LLMResponse(text=msg.get("content") or "")
        for tc in msg.get("tool_calls") or []:
            try:
                # `or {}` twice: arguments may be missing, "", or the JSON literal null
                args = json.loads(tc["function"]["arguments"] or "{}") or {}
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
