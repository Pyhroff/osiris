"""OSIRIS background daemon — task reminders + proactive routines.

Run in a second terminal:  python daemon.py  (or `osiris daemon`)

- Reminders cost zero API quota (pure DB polling).
- Routines (routines.json) run the full agent at their scheduled time once a
  day and deliver the result as a toast + Discord message + a markdown file
  in workspace/briefings/. The LLM is only built when a routine fires.
"""
import json
import time
from datetime import datetime

from osiris import config
from osiris.memory import Memory

POLL_SECONDS = 60
ROUTINES_FILE = config.ROOT / "routines.json"
BRIEFINGS_DIR = config.WORKSPACE / "briefings"


class _NoLLM:
    """Reminder path never embeds or chats — memory falls back to keyword search."""
    def embed(self, text):
        raise RuntimeError("daemon does not use the LLM for reminders")


def notify(title: str, message: str):
    try:
        from plyer import notification
        notification.notify(title=title, message=message[:250], app_name="OSIRIS", timeout=10)
    except Exception:
        print(f"\a[{datetime.now():%H:%M}] {title}: {message[:250]}")  # console fallback
    if config.DISCORD_WEBHOOK_URL:  # mirror to Discord so it reaches your phone
        try:
            import requests
            requests.post(config.DISCORD_WEBHOOK_URL,
                          json={"content": f"**{title}**\n{message[:1900]}"}, timeout=10)
        except Exception as e:
            print(f"[discord notify failed: {e}]")


def load_routines() -> list[dict]:
    if not ROUTINES_FILE.exists():
        return []
    try:
        routines = json.loads(ROUTINES_FILE.read_text(encoding="utf-8"))
        return [r for r in routines if r.get("name") and r.get("time") and r.get("prompt")]
    except (json.JSONDecodeError, OSError) as e:
        print(f"[routines.json unreadable: {e}]")
        return []


def routine_due(routine: dict, last_run: str, now: datetime) -> bool:
    """Due when its HH:MM has passed today and it hasn't run today yet. An
    optional "days" list (e.g. ["Sun"]) restricts it to those weekdays —
    omit it for the default daily cadence."""
    if last_run.startswith(now.strftime("%Y-%m-%d")):
        return False
    days = routine.get("days")
    if days and now.strftime("%a") not in days:
        return False
    return now.strftime("%H:%M") >= routine["time"]


_agent = None  # built on first routine fire, then reused


def run_routine(routine: dict, memory: Memory):
    global _agent
    if _agent is None:  # lazy: idle daemon costs zero API quota
        from osiris.agent import Agent
        from osiris.planner import Planner
        from osiris.router import ModelRouter
        from osiris.tools import build_registry
        llm = ModelRouter(verbose=False)
        mem = Memory(llm)  # LLM-backed memory for the agent (separate from reminder poller)
        _agent = Agent(llm, mem, build_registry(mem, Planner(llm, mem)))
    from osiris.agents import PROFILES
    # Routines are pre-written prompts, not live user input — declare the agent
    # explicitly rather than trusting keyword routing, which can land on a
    # profile missing tools the prompt actually needs (e.g. "mail" inside
    # "gmail_search" routing a repo+calendar+mail brief to the email-only agent).
    profile = PROFILES.get(routine.get("agent", "personal"), PROFILES["personal"])
    result = _agent.run(routine["prompt"], profile=profile)
    _agent.history.clear()  # routines are independent; don't let them pile up context

    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = BRIEFINGS_DIR / f"{datetime.now():%Y-%m-%d}-{routine['name']}.md"
    out.write_text(f"# {routine['name']} — {datetime.now():%Y-%m-%d %H:%M}\n\n{result}\n",
                   encoding="utf-8")
    notify(f"OSIRIS · {routine['name']}", result)
    print(f"[{datetime.now():%H:%M}] routine '{routine['name']}' done -> {out.name}")


def poll_once(memory: Memory, routines: list[dict]) -> list[str]:
    """One reminder+routine poll cycle. Returns a human-readable line per thing
    that fired, so a caller (the standalone daemon, or the Discord bot running
    this same cycle internally) can react beyond the toast/webhook notify()
    already does — e.g. DM you directly instead of only mirroring to a webhook."""
    fired = []
    for task in memory.due_unnotified():
        notify("OSIRIS reminder", task["description"])
        memory.mark_notified(task["id"])
        print(f"[{datetime.now():%H:%M}] notified: #{task['id']} {task['description']}")
        fired.append(f"reminder: {task['description']}")

    now = datetime.now()
    for routine in routines:
        if routine_due(routine, memory.routine_last_run(routine["name"]), now):
            memory.mark_routine_run(routine["name"])  # mark first: a crash must not retry-spam quota
            try:
                run_routine(routine, memory)
                fired.append(f"routine '{routine['name']}' completed — see workspace/briefings/")
            except Exception as e:
                print(f"[routine '{routine['name']}' failed: {e}]")
                notify(f"OSIRIS · {routine['name']} failed", str(e)[:200])
                fired.append(f"routine '{routine['name']}' failed: {e}")
    return fired


def main():
    memory = Memory(_NoLLM(), db_path=config.DB_PATH)
    routines = load_routines()
    print(f"OSIRIS daemon: {len(routines)} routine(s), reminders every {POLL_SECONDS}s. Ctrl+C to stop.")
    for r in routines:
        cadence = "/".join(r["days"]) if r.get("days") else "daily"
        print(f"  · {r['name']} at {r['time']} {cadence}")

    while True:
        poll_once(memory, routines)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ndaemon stopped.")
