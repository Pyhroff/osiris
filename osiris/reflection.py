"""Reflection pass: distill a session into durable long-term memories.

Runs at session end (REPL) and periodically in the web server. Extracted facts
are deduplicated against existing memories by embedding similarity, so
reflecting twice on the same conversation stores nothing new.
"""
from __future__ import annotations

import json
import re

from .memory import Memory
from .providers import BaseLLM

_PROMPT = """You are the reflection module of OSIRIS, a personal AI operating system.
Review this session transcript and extract durable facts about the user worth
remembering in FUTURE sessions: preferences, goals, projects, skills, people,
constraints, decisions. Ignore small talk, one-off requests, and anything
already covered by the known facts below.

Respond with ONLY a JSON array of 0-5 short standalone strings, e.g.
["prefers dark terminal-style UIs", "is preparing for X in 2027"]

Known facts (do NOT repeat these):
{known}

Session transcript:
{transcript}"""

# a candidate this similar to an existing memory is a duplicate
_DUP_THRESHOLD = 0.90
_MAX_TRANSCRIPT_CHARS = 6000


def reflect(llm: BaseLLM, memory: Memory, history: list[dict]) -> list[str]:
    """Extract & store new facts from a session. Returns what was stored."""
    convo = [f"{m['role']}: {m['content']}" for m in history
             if m["role"] in ("user", "assistant") and m.get("content")]
    if len(convo) < 3:  # nothing meaningful happened
        return []
    transcript = "\n".join(convo)[-_MAX_TRANSCRIPT_CHARS:]

    known = memory.db.execute(
        "SELECT text FROM memories ORDER BY id DESC LIMIT 15").fetchall()
    known_block = "\n".join(f"- {r['text']}" for r in known) or "(none yet)"

    resp = llm.chat([{"role": "user", "content": _PROMPT.format(
        known=known_block, transcript=transcript)}])

    stored = []
    for fact in _parse(resp.text)[:5]:
        if memory.recall(fact, top_k=1, min_score=_DUP_THRESHOLD):
            continue  # near-duplicate of something we already know
        memory.remember(fact, kind="reflection")
        stored.append(fact)
    return stored


def _parse(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text.replace("```json", "").replace("```", ""), re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
        return [i.strip() for i in items if isinstance(i, str) and i.strip()]
    except json.JSONDecodeError:
        return []
