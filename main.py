"""OSIRIS — personal AI operating system. Entry point / REPL."""
import shutil
import sys
import textwrap
import time

import requests

from osiris import config
from osiris.agent import Agent
from osiris.agents import pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.reflection import reflect
from osiris.router import ModelRouter
from osiris.tools import build_registry, _export_backup

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
  /status    health check — providers, integrations LIVE-VERIFIED (not just present), memory/goal/task counts
  /backup    export memories/goals/tasks/applications to workspace/backups/ now
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


def _check_gmail() -> str:
    """Validity check, not just presence — refreshes the cached token against
    Google's servers if it's expired but has a refresh token. Never falls
    through to the interactive browser flow (that only happens on a real
    Gmail/Calendar/Drive tool call), so /status can never pop a browser."""
    cred_file = config.ROOT / "credentials.json"
    token_file = config.ROOT / "token.json"
    if not cred_file.exists():
        return "gmail/calendar/drive: not set up — credentials.json missing"
    if not token_file.exists():
        return ("gmail/calendar/drive: credentials.json present, not yet authorized — "
                "ask anything Gmail/Calendar/Drive-related to sign in")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from osiris.google_services import SCOPES
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if creds.valid:
            return "gmail/calendar/drive: authorized and working"
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            return "gmail/calendar/drive: token refreshed and working"
        return "gmail/calendar/drive: token invalid, no refresh token — delete token.json and re-authorize"
    except Exception as e:
        return (f"gmail/calendar/drive: token expired/invalid ({type(e).__name__}) — "
               f"delete token.json and run any mail/calendar command to re-authorize")


def _check_github() -> str:
    if not config.GITHUB_TOKEN:
        return "github: not set — GITHUB_TOKEN empty"
    try:
        r = requests.get("https://api.github.com/user",
                         headers={"Authorization": f"Bearer {config.GITHUB_TOKEN}"}, timeout=5)
        if r.status_code == 200:
            return f"github: authorized as {r.json().get('login', '?')}"
        return f"github: token set but not working (HTTP {r.status_code}) — regenerate at github.com/settings/tokens"
    except requests.RequestException as e:
        return f"github: couldn't verify ({type(e).__name__})"


def _check_discord() -> str:
    if not config.DISCORD_WEBHOOK_URL:
        return "discord: not set"
    try:
        # GET on a webhook URL reads its info — doesn't post a message.
        r = requests.get(config.DISCORD_WEBHOOK_URL, timeout=5)
        return "discord: webhook valid" if r.status_code == 200 else \
            f"discord: webhook not working (HTTP {r.status_code})"
    except requests.RequestException as e:
        return f"discord: couldn't verify ({type(e).__name__})"


def _check_ollama() -> str:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=1)
        n = len(r.json().get("models", []))
        return f"ollama: reachable ({n} model{'s' if n != 1 else ''} pulled)"
    except requests.RequestException:
        return "ollama: not running (fine if it's not in your provider chain)"


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
    if config.ON_ONEDRIVE and not config.ACK_ONEDRIVE:
        print(f"{_DIM} note: this folder syncs via OneDrive — osiris.db/credentials.json/token.json/"
             f"workspace/ (all git-ignored, so they never reach GitHub) still sync to Microsoft's "
             f"cloud like everything else here. Set OSIRIS_ACK_ONEDRIVE=1 in .env to silence this.{_RESET}\n")
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
            print(_check_gmail())
            print(_check_github())
            print(f"web search: {'configured' if config.TAVILY_API_KEY else 'not set — TAVILY_API_KEY empty'}")
            print(_check_discord())
            print(_check_ollama())
            print(f"memories stored: {memory.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0]}")
            print(f"active goals: {len(memory.list_goals())}")
            print(f"pending tasks: {len(memory.list_tasks())}")
        elif line == "/backup":
            print(_export_backup(memory))
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
