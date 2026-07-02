"""OSIRIS background daemon — fires Windows toast notifications when tasks come due.

Run in a second terminal:  python daemon.py
(Only reads/writes osiris.db — no LLM calls, so it costs zero API quota.)
"""
import time
from datetime import datetime

from osiris import config
from osiris.memory import Memory

POLL_SECONDS = 60


class _NoLLM:
    """Daemon never embeds or chats — memory falls back to keyword search."""
    def embed(self, text):
        raise RuntimeError("daemon does not use the LLM")


def notify(title: str, message: str):
    try:
        from plyer import notification
        notification.notify(title=title, message=message, app_name="OSIRIS", timeout=10)
    except Exception:
        print(f"\a[{datetime.now():%H:%M}] {title}: {message}")  # console fallback
    if config.DISCORD_WEBHOOK_URL:  # mirror to Discord so reminders reach your phone
        try:
            import requests
            requests.post(config.DISCORD_WEBHOOK_URL,
                          json={"content": f"⏰ **{title}** — {message}"}, timeout=10)
        except Exception as e:
            print(f"[discord notify failed: {e}]")


def main():
    memory = Memory(_NoLLM(), db_path=config.DB_PATH)
    print(f"OSIRIS daemon watching {config.DB_PATH} (poll every {POLL_SECONDS}s, Ctrl+C to stop)")
    while True:
        for task in memory.due_unnotified():
            notify("OSIRIS reminder", task["description"])
            memory.mark_notified(task["id"])
            print(f"[{datetime.now():%H:%M}] notified: #{task['id']} {task['description']}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ndaemon stopped.")
