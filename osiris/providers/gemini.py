"""Gemini adapter (free tier) — also the default embedder for memory."""
from __future__ import annotations

import time

from google import genai
from google.genai import errors, types

from .. import config
from .base import BaseLLM, LLMResponse, ToolCall


def _to_gemini_schema(s: dict) -> types.Schema:
    """JSON-schema dict -> Gemini Schema (types are uppercase enums there)."""
    return types.Schema(
        type=s.get("type", "string").upper(),
        description=s.get("description"),
        enum=s.get("enum"),
        properties={k: _to_gemini_schema(v) for k, v in s["properties"].items()} if "properties" in s else None,
        items=_to_gemini_schema(s["items"]) if "items" in s else None,
        required=s.get("required"),
    )


def _to_gemini_contents(messages: list[dict]) -> list[types.Content]:
    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        elif m["role"] == "assistant":
            parts = []
            if m.get("content"):
                parts.append(types.Part(text=m["content"]))
            for tc in m.get("tool_calls", []):
                parts.append(types.Part(function_call=types.FunctionCall(name=tc.name, args=tc.args)))
            if parts:  # an empty model turn is invalid — skip rather than poison the history
                contents.append(types.Content(role="model", parts=parts))
        elif m["role"] == "tool":
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=m["name"], response={"result": m["content"]})],
            ))
    return contents


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, api_key: str = "", model: str = ""):
        self.client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)
        self.model = model or config.CHAT_MODEL

    def chat(self, messages, system="", tools=None):
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_to_gemini_schema(t["parameters"]) if t.get("parameters", {}).get("properties") else None,
                ) for t in tools
            ])] if tools else None,
            temperature=0.7,
        )
        resp = self._call_with_retry(_to_gemini_contents(messages), cfg)

        out = LLMResponse()
        candidate = resp.candidates[0] if resp.candidates else None
        for part in (candidate.content.parts or []) if candidate and candidate.content else []:
            if part.text:
                out.text += part.text
            if part.function_call:
                out.tool_calls.append(ToolCall(name=part.function_call.name, args=dict(part.function_call.args or {})))
        return out

    def embed(self, text: str) -> list[float]:
        resp = self.client.models.embed_content(
            model=config.EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
        )
        return list(resp.embeddings[0].values)

    def _call_with_retry(self, contents, cfg, attempts: int = 2):
        """Brief backoff on 429; if still throttled, raise so the router fails over."""
        for i in range(attempts):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
            except errors.APIError as e:
                if e.code == 429 and i < attempts - 1:
                    time.sleep(10)
                else:
                    raise
