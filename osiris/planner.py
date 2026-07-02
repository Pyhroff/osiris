"""Planning engine: long-horizon goal -> concrete milestones, persisted."""
from __future__ import annotations

import json
import re
from datetime import date

from .memory import Memory
from .providers import BaseLLM

_PROMPT = """You are the planning engine of OSIRIS, a personal AI operating system.
Today is {today}. Break the user's goal into 4-8 concrete, ordered milestones.

Respond with ONLY a JSON array, no prose:
[{{"title": "...", "detail": "one actionable sentence", "due": "YYYY-MM-DD"}}]

Due dates must be realistic and spread across the goal's natural timespan.
Goal: {goal}"""


class Planner:
    def __init__(self, llm: BaseLLM, memory: Memory):
        self.llm = llm
        self.memory = memory

    def create_plan(self, goal: str) -> dict:
        resp = self.llm.chat(
            [{"role": "user", "content": _PROMPT.format(today=date.today().isoformat(), goal=goal)}])
        milestones = self._parse(resp.text)
        if not milestones:
            return {"error": "planner returned no parseable milestones", "raw": resp.text[:500]}
        goal_id = self.memory.add_goal(goal, milestones)
        return {"goal_id": goal_id, "milestones": milestones}

    @staticmethod
    def _parse(text: str) -> list[dict]:
        match = re.search(r"\[.*\]", text.replace("```json", "").replace("```", ""), re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
            return [m for m in items if isinstance(m, dict) and m.get("title")]
        except json.JSONDecodeError:
            return []
