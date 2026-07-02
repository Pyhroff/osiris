"""OSIRIS — personal AI operating system. Entry point / REPL."""
import sys

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

HELP = """commands:
  /goals     show active goals & milestones
  /tasks     show pending scheduled tasks
  /reflect   distill this session into long-term memories now
  /help      this message
  /quit      exit (reflects automatically)
anything else goes straight to the agent."""


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
        llm = ModelRouter()
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
            line = input("you › ").strip()
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
        else:
            profile = pick_agent(line)
            if profile.key != "personal":
                print(f"  [→ {profile.key} agent]")
            try:
                echo = lambda n, a: print(f"  ⚙ {n}({', '.join(f'{k}={v!r}' for k, v in a.items())})")
                print(f"\nosiris › {agent.run(line, profile=profile, on_tool=echo)}\n")
            except Exception as e:
                print(f"\n[error] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
