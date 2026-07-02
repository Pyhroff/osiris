"""Agent runtime: the reasoning loop that turns user intent into tool calls.

Flow per turn:  route to a specialized agent profile -> recall relevant
memories -> chat with that profile's tools -> execute tool calls -> feed
results back -> repeat until the model answers in plain text (or the step
cap trips). One shared history, so context survives agent switches.
"""
from __future__ import annotations

from datetime import datetime

from . import config
from .agents import AgentProfile, pick_agent
from .memory import Memory
from .providers import BaseLLM
from .tools import ToolRegistry

_SYSTEM = """You are OSIRIS, a personal AI operating system for {user}.
{persona}

Rules:
- Use tools proactively; don't describe what you could do — do it.
- If the user just wants to talk, think, or ask questions, chat normally — tools are optional.
- Store important facts about the user with `remember` without being asked.
- When the user states a long-horizon ambition, offer to `create_plan` it.
- Be concise. Current datetime: {now}.

About your own setup (answer questions about it from here, do NOT guess):
- The user's project folders you can access: {allowed_dirs}
  Use these EXACT paths — never guess paths like C:\\Users\\...\\Desktop.
- Your configuration lives in `{root}\\.env` (NOT the user's home directory).
  Keys: GEMINI_API_KEY, GROQ_API_KEY, GITHUB_TOKEN, DISCORD_WEBHOOK_URL.
- Gmail/Calendar/Drive need `{root}\\credentials.json` (Google Cloud OAuth desktop client).
- NEVER write API keys or secrets with tools — tell the user to edit .env
  themselves and restart. Changes to .env require a restart to take effect.
{memory_block}"""


class Agent:
    def __init__(self, llm: BaseLLM, memory: Memory, tools: ToolRegistry, user: str = "the user"):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.user = user
        self.history: list[dict] = []  # session-scoped; durable facts go via `remember`

    def run(self, user_input: str, profile: AgentProfile | None = None, on_tool=None) -> str:
        profile = profile or pick_agent(user_input)
        recalled = self.memory.recall(user_input, top_k=3)
        memory_block = ("\nRelevant memories:\n" + "\n".join(f"- {m}" for m in recalled)) if recalled else ""
        system = _SYSTEM.format(user=self.user, persona=profile.persona,
                                now=datetime.now().isoformat(timespec="minutes"),
                                root=config.ROOT,
                                allowed_dirs="; ".join(str(d) for d in config.ALLOWED_DIRS),
                                memory_block=memory_block)
        specs = self.tools.specs(only=set(profile.tools) or None)

        self._trim_history()
        self.history.append({"role": "user", "content": user_input})

        for _ in range(config.MAX_AGENT_STEPS):
            resp = self.llm.chat(self.history, system=system, tools=specs)
            self.history.append({"role": "assistant", "content": resp.text or None,
                                 "tool_calls": resp.tool_calls})
            if not resp.tool_calls:
                return resp.text or "(no response)"

            for tc in resp.tool_calls:
                tc.args = tc.args or {}  # some models send null args for no-param tools
                if on_tool:
                    on_tool(tc.name, tc.args)
                result = self.tools.execute(tc.name, tc.args)
                self.history.append({"role": "tool", "name": tc.name,
                                     "tool_call_id": tc.id, "content": result})

        return "(stopped: hit max agent steps — try a narrower request)"

    def _trim_history(self):
        """Drop the oldest turns once history exceeds the cap. Cuts only at a
        user-message boundary so tool results never orphan their tool calls."""
        if len(self.history) <= config.MAX_HISTORY:
            return
        cut = len(self.history) - config.MAX_HISTORY
        while cut < len(self.history) and self.history[cut]["role"] != "user":
            cut += 1
        self.history = self.history[cut:]
