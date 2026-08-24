"""Argument repair via schema-constrained decoding — a more reliable fix for
the exact failure mode openai_compat.py's regex-based rescues already target.

Small local models routinely pick the right tool but garble the arguments:
truncated JSON, a stray comment, wrong key names, prose bleeding into the
value. openai_compat.py's `_rescue_pseudo_calls`/`_rescue_json_call` catch
some of that with regexes — best-effort, pattern-specific, and silent when
the garbling doesn't match either pattern.

This does the same job a different way: instead of pattern-matching bad
output after the fact, force a fresh completion through Ollama/llama.cpp's
grammar-constrained decoder using the tool's own JSON schema as the format —
the model literally cannot emit a token that violates the schema. Verified
empirically against Ollama 0.32.15's OpenAI-compatible endpoint (what
OpenAICompatLLM actually calls) with response_format.type="json_schema",
against qwen3, gemma3, and llama3.2 — see benchmarks/tool_calling_reliability.py
for the measured before/after numbers this is based on, not assumed.

Scope, stated plainly: this repairs ARGUMENTS for a tool the model already
decided to call. It does not fix tool SELECTION — if the model never
attempted a call when one was needed, there's no tool schema to constrain
against, and this can't help. That's a different, harder problem.
"""
from __future__ import annotations

import json

import requests


def needs_repair(tool_call_args: dict | None, schema: dict) -> bool:
    """True if args are missing required fields the tool's schema demands.
    Doesn't attempt full JSON-Schema validation (types, patterns, etc.) —
    just the cheap, high-signal check: are the required keys even present."""
    if not isinstance(tool_call_args, dict):
        return True
    required = schema.get("required", [])
    return any(key not in tool_call_args for key in required)


def repair_args(base_url: str, api_key: str, model: str,
                tool_name: str, tool_schema: dict,
                user_context: str, timeout: int = 180) -> dict | None:
    """Re-decode just this tool's arguments, constrained to its own JSON
    schema via response_format. Returns the repaired args dict, or None if
    the repair call itself failed (network error, model still couldn't
    produce valid JSON, etc.) — callers should keep whatever they had
    before, not silently substitute an empty dict."""
    prompt = (
        f"Extract the arguments for calling the tool \"{tool_name}\" from this "
        f"request. Respond with ONLY the arguments as JSON matching the schema.\n\n"
        f"Request: {user_context}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": tool_name, "schema": tool_schema},
        },
    }
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload, timeout=timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except (requests.RequestException, KeyError, json.JSONDecodeError, ValueError):
        return None
