"""OSIRIS — personal AI operating system. Entry point / REPL."""
import shutil
import sys
import textwrap
import time

from osiris import config
from osiris.agent import Agent
from osiris.agents import pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.reflection import reflect
from osiris.router import ModelRouter
from osiris.tools import build_registry

BANNER = r"""
  ___  ____ ___ ____  ___ ____
 / _ \/ ___|_ _|  _ \|_ _/ ___|
| | | \___ \| || |_) || |\___ \
| |_| |___) | ||  _ < | | ___) |
 \___/|____/___|_| \_\___|____/
 intelligence as the kernel · type /help
"""

# Real ANSI escapes, not markdown — this actually renders in a terminal,
# unlike **bold**/# headers which just show as literal punctuation. Off
# entirely when stdout isn't a real terminal (piped/redirected output stays
# clean, and my own testing via a pipe doesn't get escape-code noise).
_TTY = sys.stdout.isatty()
_DIM = "\033[90m" if _TTY else ""
_AMBER = "\033[38;5;215m" if _TTY else ""
_RESET = "\033[0m" if _TTY else ""


def _wrap(text: str) -> str:
    """Wrap to the real terminal width with a hanging indent under "- "
    bullets, so a wrapped bullet's continuation lines align under the text
    instead of starting back at column 0 and looking like a new item."""
    width = max(40, min(shutil.get_terminal_size(fallback=(100, 24)).columns - 2, 100))
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        hang = "  " if para.lstrip().startswith("- ") else ""
        out.append(textwrap.fill(para, width=width, subsequent_indent=hang))
    return "\n".join(out)

HELP = """commands:
  /goals     show active goals & milestones
  /tasks     show pending scheduled tasks
  /status    health check — providers, integrations configured, memory/goal/task counts
  /model     which provider answered last + who's cooling down right now
  /loop <seconds> <prompt>   repeat a prompt on an interval until Ctrl+C
  /reflect   distill this session into long-term memories now
  /clear     wipe this session's conversation (long-term memories are kept)
  /help      this message
  /quit      exit (reflects automatically)
anything else goes straight to the agent.

if something breaks mid-task or a reply looks wrong/stuck:
  say "continue" to resume with a fresh step budget, or /clear to reset the
  conversation and start over — neither loses anything you've told it to remember."""


def _run_reflection(llm, memory, agent):
    try:
        stored = reflect(llm, memory, agent.history)
    except Exception as e:
        print(f"[reflection failed: {e}]")
        return
    if stored:
        print("reflected — remembered:")
        for f in stored:
            print(f"  + {f}")
    else:
        print("reflected — nothing new worth keeping.")


def main():
    try:
        llm = ModelRouter(verbose=config.DEBUG)
    except RuntimeError as e:
        sys.exit(str(e))

    memory = Memory(llm)
    planner = Planner(llm, memory)
    agent = Agent(llm, memory, build_registry(memory, planner))

    print(BANNER)
    print(f" providers: {llm.describe()}\n")
    due = memory.due_tasks()
    if due:
        print("⏰ due now:")
        for t in due:
            print(f"   #{t['id']} {t['description']} (due {t['due']})")
        print()

    while True:
        try:
            line = input(f"{_AMBER}you › {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            if len(agent.history) >= 3:
                _run_reflection(llm, memory, agent)
            print("bye.")
            break
        if line == "/help":
            print(HELP)
        elif line == "/reflect":
            _run_reflection(llm, memory, agent)
        elif line == "/clear":
            agent.history.clear()
            print("conversation cleared — long-term memories are untouched.")
        elif line == "/goals":
            goals = memory.list_goals()
            if not goals:
                print("no active goals — tell me an ambition and I'll plan it.")
            for g in goals:
                print(f"◆ [{g['id']}] {g['title']}")
                for m in g["milestones"]:
                    mark = "✓" if m["status"] == "done" else "·"
                    print(f"   {mark} [{m['id']}] {m['title']} (due {m['due']})")
        elif line == "/tasks":
            tasks = memory.list_tasks()
            if not tasks:
                print("no pending tasks.")
            for t in tasks:
                print(f"· [#{t['id']}] {t['description']} — due {t['due']}")
        elif line == "/model":
            print(f"chain: {llm.describe()}")
            now = time.time()
            cooling = [f"{n} ({int(until - now)}s left)" for n, until in llm._cooldown.items() if until > now]
            print("cooling down: " + ", ".join(cooling) if cooling else "cooling down: none")
            via = getattr(llm, "active", "")
            print(f"last answered by: {via or '(nothing sent yet this session)'}")
        elif line == "/status":
            print(f"providers: {llm.describe()}")
            print(f"gmail/calendar/drive: {'configured' if (config.ROOT / 'credentials.json').exists() else 'not set up — credentials.json missing'}")
            print(f"github: {'configured' if config.GITHUB_TOKEN else 'not set — GITHUB_TOKEN empty'}")
            print(f"web search: {'configured' if config.TAVILY_API_KEY else 'not set — TAVILY_API_KEY empty'}")
            print(f"discord: {'configured' if config.DISCORD_WEBHOOK_URL else 'not set'}")
            print(f"memories stored: {memory.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0]}")
            print(f"active goals: {len(memory.list_goals())}")
            print(f"pending tasks: {len(memory.list_tasks())}")
        elif line == "/loop" or line.startswith("/loop "):
            rest = line[len("/loop"):].strip().split(" ", 1)
            if len(rest) < 2 or not rest[0].isdigit():
                print("usage: /loop <seconds> <prompt>")
            else:
                interval, prompt = int(rest[0]), rest[1]
                profile = pick_agent(prompt)
                print(f"looping every {interval}s — Ctrl+C to stop")
                try:
                    while True:
                        try:
                            reply = agent.run(prompt, profile=profile)
                            via = getattr(llm, "active", "")
                            print(f"\nosiris › {_wrap(reply)}" + (f"\n  {_DIM}[{via}]{_RESET}" if via else ""))
                        except Exception as e:
                            print(f"\n[error] {type(e).__name__}: {e}")
                        time.sleep(interval)
                except KeyboardInterrupt:
                    print("\nloop stopped.")
        else:
            profile = pick_agent(line)
            if profile.key != "personal":
                print(f"  {_DIM}[→ {profile.key} agent]{_RESET}")
            try:
                echo = lambda n, a: print(
                    f"  {_DIM}⚙ {n}({', '.join(f'{k}={v!r}' for k, v in a.items())}){_RESET}")
                reply = agent.run(line, profile=profile, on_tool=echo)
                via = getattr(llm, "active", "")
                print(f"\nosiris › {_wrap(reply)}" + (f"\n  {_DIM}[{via}]{_RESET}\n" if via else "\n"))
            except Exception as e:
                print(f"\n[error] {type(e).__name__}: {e}")
                print("  if this keeps happening, try /clear to reset the conversation\n")


if __name__ == "__main__":
    main()
