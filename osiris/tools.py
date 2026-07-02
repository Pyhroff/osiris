"""Tool registry + the built-in toolset the agent runtime can call.

Tool params are plain JSON schema — provider-neutral; the LLM adapter
converts them to whatever the underlying API wants.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests

from . import config
from .memory import Memory
from .planner import Planner


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        self._tools[name] = {"name": name, "description": description,
                             "parameters": parameters, "fn": fn}

    def specs(self, only: set[str] | None = None) -> list[dict]:
        return [{k: t[k] for k in ("name", "description", "parameters")}
                for t in self._tools.values() if only is None or t["name"] in only]

    def execute(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"error: unknown tool '{name}'"
        try:
            result = self._tools[name]["fn"](**args)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


def _safe_path(rel: str) -> Path:
    """Resolve a path inside the sandbox; refuse anything that escapes it."""
    p = (config.WORKSPACE / rel).resolve()
    if not p.is_relative_to(config.WORKSPACE.resolve()):
        raise PermissionError(f"path escapes workspace: {rel}")
    return p


def _allowed_path(p: str) -> Path:
    """Resolve a user path; must land inside one of the allowed roots (Desktop by default)."""
    path = Path(p)
    if not path.is_absolute():
        for root in config.ALLOWED_DIRS:
            if (root / p).exists():
                path = root / p
                break
        else:
            path = config.ALLOWED_DIRS[0] / p
    path = path.resolve()
    if not any(path.is_relative_to(r) for r in config.ALLOWED_DIRS):
        raise PermissionError(f"path outside allowed directories: {p}")
    return path


def _git(repo: str, *args: str) -> str:
    repo_path = _allowed_path(repo)
    cmd = ["git", "-C", str(repo_path)]
    if config.GIT_NAME:
        cmd += ["-c", f"user.name={config.GIT_NAME}"]
    if config.GIT_EMAIL:
        cmd += ["-c", f"user.email={config.GIT_EMAIL}"]
    r = subprocess.run(cmd + list(args), capture_output=True, text=True, timeout=120)
    out = (r.stdout + r.stderr).strip()
    return out or ("ok" if r.returncode == 0 else f"git exited {r.returncode}")


def _gh_headers() -> dict:
    if not config.GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set in .env — create one at github.com/settings/tokens")
    return {"Authorization": f"Bearer {config.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json"}


def build_registry(memory: Memory, planner: Planner) -> ToolRegistry:
    reg = ToolRegistry()

    # -- memory ----------------------------------------------------------------
    reg.register(
        "remember", "Store a durable fact about the user or their world for future sessions.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "the fact to store"}},
         "required": ["text"]},
        lambda text: f"stored memory #{memory.remember(text)}")

    reg.register(
        "recall", "Search stored memories for facts relevant to a query.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        lambda query: json.dumps(memory.recall(query, top_k=5)) or "no relevant memories")

    # -- planning ----------------------------------------------------------------
    reg.register(
        "create_plan", "Break a long-horizon goal into milestones with due dates and persist it.",
        {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]},
        lambda goal: planner.create_plan(goal))

    reg.register(
        "list_goals", "List all active goals and their milestone progress.",
        {"type": "object", "properties": {}},
        lambda: memory.list_goals() or "no active goals")

    reg.register(
        "complete_milestone", "Mark a milestone as done by its id.",
        {"type": "object", "properties": {"milestone_id": {"type": "integer"}}, "required": ["milestone_id"]},
        lambda milestone_id: "done" if memory.complete_milestone(milestone_id) else "error: no such milestone")

    # -- scheduler ----------------------------------------------------------------
    reg.register(
        "schedule_task", "Schedule a reminder/task for a specific date-time (ISO format).",
        {"type": "object",
         "properties": {"description": {"type": "string"},
                        "due": {"type": "string", "description": "ISO datetime e.g. 2026-07-05T09:00:00"}},
         "required": ["description", "due"]},
        lambda description, due: f"task #{memory.add_task(description, due)} scheduled for {due}")

    reg.register(
        "list_tasks", "List pending scheduled tasks.",
        {"type": "object", "properties": {}},
        lambda: memory.list_tasks() or "no pending tasks")

    reg.register(
        "complete_task", "Mark a scheduled task as done by its id.",
        {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]},
        lambda task_id: "done" if memory.complete_task(task_id) else "error: no such task")

    # -- filesystem (sandboxed to ./workspace) -------------------------------------
    reg.register(
        "write_file", "Write text to a file inside the OSIRIS workspace directory.",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "relative path e.g. notes/ideas.md"},
                        "content": {"type": "string"}},
         "required": ["path", "content"]},
        lambda path, content: _write_file(path, content))

    reg.register(
        "read_file", "Read a text file from the OSIRIS workspace directory.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        lambda path: _safe_path(path).read_text(encoding="utf-8"))

    reg.register(
        "list_files", "List files in the OSIRIS workspace directory.",
        {"type": "object", "properties": {}},
        lambda: [str(p.relative_to(config.WORKSPACE)) for p in config.WORKSPACE.rglob("*") if p.is_file()]
        or "workspace is empty")

    # -- desktop filesystem (read-only, limited to allowed roots) -----------------
    reg.register(
        "fs_list", "List a folder on the user's computer (Desktop and subfolders by default).",
        {"type": "object", "properties": {"path": {"type": "string", "description": "e.g. 'StylesOS' or an absolute path"}},
         "required": ["path"]},
        lambda path: _fs_list(path))

    reg.register(
        "fs_read", "Read a text file from the user's computer (within allowed folders).",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        lambda path: _fs_read(path))

    reg.register(
        "fs_find", "Find files by glob pattern under an allowed folder (e.g. pattern '**/*.py').",
        {"type": "object",
         "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "folder to search; defaults to Desktop"}},
         "required": ["pattern"]},
        lambda pattern, path="": _fs_find(pattern, path))

    # -- local git (repos under allowed roots) -------------------------------------
    reg.register(
        "git_status", "Show git status of a local repository folder.",
        {"type": "object", "properties": {"repo": {"type": "string", "description": "repo folder, e.g. 'PromptStrike'"}},
         "required": ["repo"]},
        lambda repo: _git(repo, "status", "--short", "--branch"))

    reg.register(
        "git_diff", "Show uncommitted changes in a local repository.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _git(repo, "diff", "--stat", "HEAD")[:6000])

    reg.register(
        "git_log", "Show recent commits in a local repository.",
        {"type": "object",
         "properties": {"repo": {"type": "string"}, "limit": {"type": "integer", "description": "default 5"}},
         "required": ["repo"]},
        lambda repo, limit=5: _git(repo, "log", f"-{limit}", "--oneline"))

    reg.register(
        "git_commit", "Stage all changes and commit in a local repository. Show the user the status first and confirm the message.",
        {"type": "object",
         "properties": {"repo": {"type": "string"}, "message": {"type": "string"}},
         "required": ["repo", "message"]},
        lambda repo, message: _git(repo, "add", "-A") + "\n" + _git(repo, "commit", "-m", message))

    reg.register(
        "git_push", "Push commits to the remote. Always confirm with the user before pushing.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _git(repo, "push"))

    reg.register(
        "git_sync_check",
        "Check whether a local repo matches its GitHub remote: fetches, then reports "
        "uncommitted local changes and commits ahead/behind the remote. Use this before "
        "deciding whether to commit and push.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _git_sync_check(repo))

    # -- github ----------------------------------------------------------------
    reg.register(
        "github_list_issues", "List open issues on a GitHub repo (format: owner/repo).",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _gh_issues(repo))

    reg.register(
        "github_create_issue", "Create an issue on a GitHub repo you own (format: owner/repo).",
        {"type": "object",
         "properties": {"repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}},
         "required": ["repo", "title"]},
        lambda repo, title, body="": _gh_create_issue(repo, title, body))

    # -- gmail + calendar (OAuth; see osiris/google_services.py for setup) --------
    from . import google_services as g

    reg.register(
        "gmail_search", "Search the user's Gmail inbox (standard Gmail query syntax, e.g. 'is:unread', 'from:foo@bar.com').",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "max_results": {"type": "integer", "description": "default 5"}},
         "required": ["query"]},
        lambda query, max_results=5: g.gmail_search(query, max_results) or "no matching emails")

    reg.register(
        "gmail_read", "Read the full body of an email by its message id (from gmail_search).",
        {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]},
        lambda message_id: g.gmail_read(message_id))

    reg.register(
        "gmail_send", "Send an email from the user's Gmail account. Confirm with the user before sending.",
        {"type": "object",
         "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "subject", "body"]},
        lambda to, subject, body: g.gmail_send(to, subject, body))

    reg.register(
        "gmail_create_draft",
        "Save an email as a draft in the user's Gmail Drafts folder (nothing is sent). "
        "For multiple drafts, call once per email.",
        {"type": "object",
         "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "subject", "body"]},
        lambda to, subject, body: g.gmail_create_draft(to, subject, body))

    reg.register(
        "calendar_list_events", "List upcoming Google Calendar events.",
        {"type": "object",
         "properties": {"days_ahead": {"type": "integer", "description": "default 7"}}},
        lambda days_ahead=7: g.calendar_list_events(days_ahead) or "no upcoming events")

    reg.register(
        "calendar_create_event", "Create a Google Calendar event (local timezone).",
        {"type": "object",
         "properties": {"title": {"type": "string"},
                        "start_iso": {"type": "string", "description": "e.g. 2026-07-05T09:00:00"},
                        "end_iso": {"type": "string"},
                        "description": {"type": "string"}},
         "required": ["title", "start_iso", "end_iso"]},
        lambda title, start_iso, end_iso, description="": g.calendar_create_event(title, start_iso, end_iso, description))

    # -- google drive ----------------------------------------------------------
    reg.register(
        "drive_search", "Search the user's Google Drive by name or content.",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "max_results": {"type": "integer", "description": "default 10"}},
         "required": ["query"]},
        lambda query, max_results=10: g.drive_search(query, max_results) or "no matching files")

    reg.register(
        "drive_read", "Read the text content of a Google Drive file by its id (from drive_search). Handles Docs/Sheets/Slides and plain files.",
        {"type": "object", "properties": {"file_id": {"type": "string"}}, "required": ["file_id"]},
        lambda file_id: g.drive_read(file_id))

    # -- discord ----------------------------------------------------------------
    reg.register(
        "discord_send", "Post a message to the user's Discord channel (via webhook).",
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
        lambda message: _discord_send(message))

    # -- misc ----------------------------------------------------------------
    reg.register(
        "current_datetime", "Get the current local date and time.",
        {"type": "object", "properties": {}},
        lambda: datetime.now().isoformat(timespec="seconds"))

    return reg


def _write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


def _git_sync_check(repo: str) -> dict:
    fetch = _git(repo, "fetch")
    branch = _git(repo, "status", "-sb")           # "## main...origin/main [ahead 2]"
    changes = _git(repo, "status", "--porcelain")
    if changes == "ok":  # _git substitutes "ok" for empty output — clean tree here
        changes = ""
    return {
        "fetch": fetch if "error" in fetch.lower() or "fatal" in fetch.lower() else "ok",
        "branch_vs_remote": branch.splitlines()[0] if branch else branch,
        "uncommitted_files": changes.splitlines() if changes and "fatal" not in changes else [],
        "in_sync": not changes.strip() and "ahead" not in branch and "behind" not in branch,
    }


def _fs_list(path: str) -> list[str] | str:
    p = _allowed_path(path)
    if not p.is_dir():
        return f"error: not a folder: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    return [f"{'[dir] ' if e.is_dir() else ''}{e.name}" for e in entries[:100]] or "empty folder"


def _fs_read(path: str, max_chars: int = 8000) -> str:
    p = _allowed_path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n…(truncated)" if len(text) > max_chars else "")


def _fs_find(pattern: str, path: str = "") -> list[str] | str:
    root = _allowed_path(path) if path else config.ALLOWED_DIRS[0]
    hits = []
    for f in root.rglob(pattern):
        if any(part.startswith(".") or part in ("node_modules", "__pycache__") for part in f.parts):
            continue
        hits.append(str(f))
        if len(hits) >= 50:
            break
    return hits or "no matches"


def _discord_send(message: str) -> str:
    if not config.DISCORD_WEBHOOK_URL:
        raise RuntimeError("DISCORD_WEBHOOK_URL not set in .env — in Discord: channel settings "
                           "→ Integrations → Webhooks → New Webhook → copy URL")
    r = requests.post(config.DISCORD_WEBHOOK_URL, json={"content": message[:2000]}, timeout=15)
    r.raise_for_status()
    return "posted to Discord"


def _gh_issues(repo: str) -> list[dict] | str:
    r = requests.get(f"https://api.github.com/repos/{repo}/issues",
                     headers=_gh_headers(), params={"state": "open", "per_page": 10}, timeout=15)
    r.raise_for_status()
    issues = [{"number": i["number"], "title": i["title"], "url": i["html_url"]}
              for i in r.json() if "pull_request" not in i]
    return issues or "no open issues"


def _gh_create_issue(repo: str, title: str, body: str) -> str:
    r = requests.post(f"https://api.github.com/repos/{repo}/issues",
                      headers=_gh_headers(), json={"title": title, "body": body}, timeout=15)
    r.raise_for_status()
    return f"created issue #{r.json()['number']}: {r.json()['html_url']}"
