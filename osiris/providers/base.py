"""Neutral LLM contract — every provider adapter implements this.

Neutral message format (what the agent speaks):
    {"role": "user",      "content": str}
    {"role": "assistant", "content": str|None, "tool_calls": [ToolCall]}
    {"role": "tool",      "name": str, "tool_call_id": str, "content": str}
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""  # OpenAI-style providers need call ids; Gemini ignores them


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseLLM:
    name: str = "base"

    def chat(self, messages: list[dict], system: str = "", tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError
