"""Agent runtime: the reasoning loop that turns user intent into tool calls.

Flow per turn:  route to a specialized agent profile -> recall relevant
memories -> chat with that profile's tools -> execute tool calls -> feed
results back -> repeat until the model answers in plain text (or the step
cap trips). One shared history, so context survives agent switches.
"""
from __future__ import annotations

import re
from datetime import datetime

from . import config
from .agents import AgentProfile, pick_agent
from .memory import Memory
from .providers import BaseLLM
from .tools import ToolRegistry

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_UNDERSCORE_BOLD = re.compile(r"__(.+?)__")
_ITALIC = re.compile(r"\*(.+?)\*")   # run only after _BOLD, so no ** pairs remain to confuse it
_HEADER = re.compile(r"^#{1,6}\s+", flags=re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """None of OSIRIS's interfaces render markdown, so **bold**/*italic*/# headers
    show as literal punctuation, not formatting. Strip them at the source rather
    than trust every provider to follow the plain-text instruction — Groq's
    oss-120b, in particular, keeps using ** and single-* emphasis despite being
    told not to (verified live: "*Goal*:" survived until this was added)."""
    text = _BOLD.sub(r"\1", text)
    text = _UNDERSCORE_BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _HEADER.sub("", text)
    return text

_SYSTEM = """You are OSIRIS, a personal AI operating system for {user}.
{user_profile}{persona}

Rules:
- Use tools proactively; don't describe what you could do — do it.
- If the user just wants to talk, think, or ask questions, chat normally — tools are optional.
- Store important facts about the user with `remember` without being asked.
- If the user asks you to forget something, list_memories to find it, confirm which one you
  mean if there's any ambiguity, then forget it by id. It's permanent — never guess the id.
- When the user states a long-horizon ambition, offer to `create_plan` it.
- Be concise. Current datetime: {now}.
- Before acting: silently work through what the user actually needs, which tool(s) in what
  order, and what could go wrong — then act. Never show this reasoning. Never prefix a reply
  with "Let me think", "First, I will", or similar — output only the final tool calls or the
  final answer.
- Output formatting: plain text only. NEVER use **bold**, # headers, or markdown tables — none
  of your interfaces (terminal, web UI) render markdown, so it shows as literal asterisks and
  hashes, not formatting. For a list, use a plain "-" per line, sparingly. Prefer a short,
  natural, conversational answer over a labeled "report" with restated fields — if the user
  asks "tell me about myself," answer like a person would, don't dump their profile back as a
  formal structured summary unless they specifically asked for one.

About your own setup (answer questions about it from here, do NOT guess):
- The user's project folders you can access: {allowed_dirs}
  Use these EXACT paths — never guess paths like C:\\Users\\...\\Desktop.
- Your configuration lives in `{root}\\.env` (NOT the user's home directory).
  Provider keys (need at least one): GEMINI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY,
  MISTRAL_API_KEY, OPENROUTER_API_KEY, GITHUB_MODELS_TOKEN, OPENAI_API_KEY, or a
  locally running Ollama (no key). Integration keys: GITHUB_TOKEN, TAVILY_API_KEY,
  DISCORD_WEBHOOK_URL, DISCORD_BOT_TOKEN. There are more tuning knobs than this
  (models, allowed dirs, recon targets, etc.) — if asked for the exact/full list,
  fs_read `{root}\\.env.example` rather than guessing from this list, which is
  deliberately the common ones, not exhaustive.
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
        user_profile = f"About {self.user}: {config.USER_PROFILE}\n" if config.USER_PROFILE else ""
        system = _SYSTEM.format(user=self.user, persona=profile.persona, user_profile=user_profile,
                                now=datetime.now().isoformat(timespec="minutes"),
                                root=config.ROOT,
                                allowed_dirs="; ".join(str(d) for d in config.ALLOWED_DIRS),
                                memory_block=memory_block)
        specs = self.tools.specs(only=set(profile.tools) or None)

        self._trim_history()
        self.history.append({"role": "user", "content": user_input})

        last_tool_names, streak = None, 0
        for _ in range(profile.max_steps or config.MAX_AGENT_STEPS):
            resp = self.llm.chat(self.history, system=system, tools=specs)
            self.history.append({"role": "assistant", "content": resp.text or None,
                                 "tool_calls": resp.tool_calls})
            if not resp.tool_calls:
                if resp.text:
                    return _strip_markdown(resp.text)
                # Truly empty response (no text, no tool call) — observed live with
                # Gemini on casual/typo'd mail requests. Don't surrender the whole
                # turn on this; nudge and let the loop retry with the remaining
                # step budget instead of showing a dead-end "(no response)".
                self.history.append({"role": "user", "content":
                    "That produced no reply and no tool call. Answer the question directly, "
                    "or call whichever tool actually addresses it."})
                continue

            for tc in resp.tool_calls:
                tc.args = tc.args or {}  # some models send null args for no-param tools
                if on_tool:
                    on_tool(tc.name, tc.args)
                result = self.tools.execute(tc.name, tc.args)
                self.history.append({"role": "tool", "name": tc.name,
                                     "tool_call_id": tc.id, "content": result})

            # Circuit breaker: a weak model can retry the same tool call over and
            # over on a stubborn error, burning the whole step budget without
            # making progress (observed live with Mistral on a dependency issue).
            # After a few repeats, force a decision instead of another blind retry.
            tool_names = tuple(sorted(tc.name for tc in resp.tool_calls))
            streak = streak + 1 if tool_names == last_tool_names else 0
            last_tool_names = tool_names
            if streak == 3:
                self.history.append({"role": "user", "content":
                    f"You've called {', '.join(tool_names)} several times in a row without "
                    "resolving this. Stop retrying the same way — read the actual last error "
                    "carefully and make ONE different, targeted fix (not another variation of "
                    "what you already tried), or if you genuinely cannot resolve it, stop now "
                    "and say so plainly with the last error shown, instead of continuing to retry."})

        return ("(stopped: hit the step limit while working on this — say 'continue' to "
                "resume with a fresh step budget, or try a narrower request)")

    def _trim_history(self):
        """Drop the oldest turns once history exceeds the cap. Cuts only at a
        user-message boundary so tool results never orphan their tool calls."""
        if len(self.history) <= config.MAX_HISTORY:
            return
        cut = len(self.history) - config.MAX_HISTORY
        while cut < len(self.history) and self.history[cut]["role"] != "user":
            cut += 1
        self.history = self.history[cut:]
