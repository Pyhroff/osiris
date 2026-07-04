"""Tool registry + the built-in toolset the agent runtime can call.

Tool params are plain JSON schema — provider-neutral; the LLM adapter
converts them to whatever the underlying API wants.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests

from . import config
from .memory import Memory
from .planner import Planner

# Recon tools OSIRIS is allowed to shell out to, and the flags each one may
# receive. Deliberately excludes exploitation/brute-force tools (sqlmap,
# hydra, metasploit) and any nmap/curl flag that writes files or runs
# arbitrary scripts (-oN, --script, -e) — this is recon-only, not an
# exploitation framework.
_RECON_TOOLS = {
    "nmap": {"safe_flags": {"-sV", "-sC", "-sT", "-sU", "-sS", "-A", "-Pn",
                            "-T0", "-T1", "-T2", "-T3", "-T4", "-T5",
                            "--open", "-p", "--top-ports", "-6"}},
    "whois": {"safe_flags": set()},
    "dig": {"safe_flags": {"+short", "+trace", "-x", "A", "AAAA", "MX", "TXT", "NS", "CNAME"}},
    "nslookup": {"safe_flags": set()},
    "ping": {"safe_flags": {"-n", "-c"}},
    "curl": {"safe_flags": {"-I", "--head", "-v", "-s", "-L"}},
}

# Test/build/lint commands run_project_command may invoke, one repo at a time.
# Deliberately narrow — no arbitrary script execution, just the handful of
# commands you'd run by hand while checking a project's health.
_PROJECT_COMMANDS = {
    "pytest": ["pytest"],
    "npm-test": ["npm", "test"],
    "npm-build": ["npm", "run", "build"],
    "npm-lint": ["npm", "run", "lint"],
    "ruff": ["ruff", "check", "."],
    "mypy": ["mypy", "."],
}
_ARG_PATTERN = re.compile(r"[A-Za-z0-9_./:=,\-]+")


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
    expanded = Path(rel).expanduser()  # "~/..." must resolve (and get blocked), not become a literal dir
    p = (config.WORKSPACE / expanded).resolve()
    if not p.is_relative_to(config.WORKSPACE.resolve()):
        raise PermissionError(f"path escapes workspace: {rel}")
    return p


def _allowed_path(p: str) -> Path:
    """Resolve a user path; must land inside one of the allowed roots (Desktop by default)."""
    path = Path(p).expanduser()
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
        "remember", "Persist a durable fact about the user for future sessions.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "the fact to store"}},
         "required": ["text"]},
        lambda text: f"stored memory #{memory.remember(text)}")

    reg.register(
        "recall", "Search stored memories by query.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        lambda query: json.dumps(memory.recall(query, top_k=5)) or "no relevant memories")

    reg.register(
        "list_memories", "List stored memories with their ids, most recent first. Use this "
                         "to find the right id before calling forget.",
        {"type": "object", "properties": {"limit": {"type": "integer", "description": "default 50"}}},
        lambda limit=50: memory.list_memories(limit) or "no memories stored")

    reg.register(
        "forget", "Permanently delete a memory by its id (from list_memories or recall). "
                  "This cannot be undone — always confirm with the user which memory and "
                  "get an explicit go-ahead before calling this.",
        {"type": "object", "properties": {"memory_id": {"type": "integer"}}, "required": ["memory_id"]},
        lambda memory_id: "forgotten" if memory.forget(memory_id) else "error: no such memory")

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
        "fs_list", "List a folder on the user's Desktop/project folders.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "e.g. 'StylesOS' or an absolute path"}},
         "required": ["path"]},
        lambda path: _fs_list(path))

    reg.register(
        "fs_read", "Read a file from the user's Desktop/project folders.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        lambda path: _fs_read(path))

    reg.register(
        "read_pdf", "Extract the text of a PDF on the user's Desktop/project folders — resumes, "
                    "job descriptions, papers, coursework. Use before tailoring outreach to a JD "
                    "or summarizing a document.",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "e.g. 'Resume.pdf' or a full path"},
                        "max_pages": {"type": "integer", "description": "default 20"}},
         "required": ["path"]},
        lambda path, max_pages=20: _read_pdf(path, max_pages))

    reg.register(
        "note", "Quick-capture a note/idea/TIL into today's journal (workspace/journal/"
                "YYYY-MM-DD.md). Appends with a timestamp — use whenever the user says 'note "
                "this', 'jot down', 'remember for today', or captures a quick thought.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: _note(text))

    reg.register(
        "read_journal", "Read a day's captured notes (default today). Date format YYYY-MM-DD.",
        {"type": "object", "properties": {"date": {"type": "string", "description": "default today"}}},
        lambda date="": _read_journal(date))

    reg.register(
        "fs_find", "Find files by glob pattern under a Desktop folder (e.g. '**/*.py').",
        {"type": "object",
         "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "folder to search; defaults to Desktop"}},
         "required": ["pattern"]},
        lambda pattern, path="": _fs_find(pattern, path))

    reg.register(
        "fs_write", "Write or overwrite a file in Desktop project folders. Show user the content and get OK first.",
        {"type": "object",
         "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path", "content"]},
        lambda path, content: _fs_write(path, content))

    reg.register(
        "fs_edit", "Find-and-replace an exact snippet in a project file. Show user the change and get OK first.",
        {"type": "object",
         "properties": {"path": {"type": "string"},
                        "find": {"type": "string", "description": "exact text to replace"},
                        "replace": {"type": "string"}},
         "required": ["path", "find", "replace"]},
        lambda path, find, replace: _fs_edit(path, find, replace))

    # -- project scaffolding ----------------------------------------------------
    reg.register(
        "scaffold_project",
        "Write an entire project in ONE call: every file at once (architecture doc, source, "
        "config, tests). Replaces many fs_write calls. Code must be complete — no stubs or TODOs.",
        {"type": "object",
         "properties": {
             "folder": {"type": "string",
                        "description": "project folder, e.g. 'OSIRIS FILES/my-app'"},
             "files": {"type": "array", "description": "every file in the project",
                       "items": {"type": "object",
                                 "properties": {"path": {"type": "string",
                                                         "description": "relative path e.g. 'src/main.py'"},
                                                "content": {"type": "string"}},
                                 "required": ["path", "content"]}}},
         "required": ["folder", "files"]},
        lambda folder, files: _scaffold_project(folder, files))

    # -- local git (repos under allowed roots) -------------------------------------
    reg.register(
        "git_list_repos", "List every local git repo with branch + uncommitted-change count in one call.",
        {"type": "object", "properties": {}},
        lambda: _git_list_repos())

    reg.register(
        "git_init", "Create a brand-new local git repository (makes the folder if needed).",
        {"type": "object",
         "properties": {"path": {"type": "string", "description": "new repo folder, e.g. 'Test'"}},
         "required": ["path"]},
        lambda path: _git_init(path))

    reg.register(
        "git_status", "Show git status of a local repository folder.",
        {"type": "object", "properties": {"repo": {"type": "string", "description": "repo folder, e.g. 'PromptStrike'"}},
         "required": ["repo"]},
        lambda repo: _git(repo, "status", "--short", "--branch"))

    reg.register(
        "git_diff", "Show uncommitted changes in a local repository.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _git_diff(repo))

    reg.register(
        "git_log", "Show recent commits in a local repository.",
        {"type": "object",
         "properties": {"repo": {"type": "string"}, "limit": {"type": "integer", "description": "default 5"}},
         "required": ["repo"]},
        lambda repo, limit=5: _git(repo, "log", f"-{limit}", "--oneline"))

    reg.register(
        "git_commit", "Stage all changes and commit. Confirm message with user first.",
        {"type": "object",
         "properties": {"repo": {"type": "string"}, "message": {"type": "string"}},
         "required": ["repo", "message"]},
        lambda repo, message: _git(repo, "add", "-A") + "\n" + _git(repo, "commit", "-m", message))

    reg.register(
        "git_push", "Push commits to the remote origin. Confirm with user before calling.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _git_push(repo))

    reg.register(
        "git_sync_check",
        "Fetch + compare local repo against remote; reports uncommitted changes and ahead/behind status.",
        {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        lambda repo: _git_sync_check(repo))

    # -- github ----------------------------------------------------------------
    reg.register(
        "github_create_repo",
        "Create a GitHub repo from a local folder and push it (private by default). Confirm with user first.",
        {"type": "object",
         "properties": {"repo": {"type": "string", "description": "local repo folder"},
                        "name": {"type": "string", "description": "GitHub repo name; defaults to folder name"},
                        "private": {"type": "boolean", "description": "default true"}},
         "required": ["repo"]},
        lambda repo, name="", private=True: _gh_create_repo(repo, name, private))

    reg.register(
        "github_list_repos", "List the user's GitHub repos (name, visibility, last push). Returns 20 most recent.",
        {"type": "object", "properties": {}},
        lambda: _gh_list_repos())

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

    reg.register(
        "github_create_pr",
        "Open a pull request on GitHub from the current branch. Confirm with user first.",
        {"type": "object",
         "properties": {"repo": {"type": "string", "description": "local repo folder"},
                        "title": {"type": "string"},
                        "body": {"type": "string", "description": "PR description"},
                        "base": {"type": "string", "description": "target branch, default 'main'"}},
         "required": ["repo", "title"]},
        lambda repo, title, body="", base="main": _gh_create_pr(repo, title, body, base))

    # -- recon (read-only network tools, authorized targets only) -----------------
    reg.register(
        "run_recon",
        "Run a read-only recon tool (nmap/whois/dig/nslookup/ping/curl) against an "
        "authorized target. Target must be localhost/a private IP, or listed in "
        "OSIRIS_AUTHORIZED_TARGETS in .env — never run this against a target without "
        "confirmed authorization. Flags are restricted to a safe allowlist per tool "
        "(no file-writing or script-execution flags). ALWAYS show the user the exact "
        "command and get explicit confirmation before calling this.",
        {"type": "object",
         "properties": {
             "tool": {"type": "string", "description": f"one of: {', '.join(_RECON_TOOLS)}"},
             "target": {"type": "string", "description": "hostname or IP"},
             "flags": {"type": "array", "items": {"type": "string"},
                       "description": "optional flags, e.g. ['-sV', '-Pn']"},
         },
         "required": ["tool", "target"]},
        lambda tool, target, flags=None: _run_recon(tool, target, flags))

    reg.register(
        "run_project_command",
        "Run a test/build/lint command in a project repo (pytest, npm-test, npm-build, "
        "npm-lint, ruff, mypy). Runs real code and can have side effects (test artifacts, "
        "build output) — show the user the exact command and get their OK first.",
        {"type": "object",
         "properties": {
             "repo": {"type": "string", "description": "local repo folder, e.g. 'PromptStrike'"},
             "command": {"type": "string", "description": f"one of: {', '.join(_PROJECT_COMMANDS)}"},
             "extra_args": {"type": "array", "items": {"type": "string"},
                            "description": "optional extra args, e.g. ['-k', 'test_auth']"},
         },
         "required": ["repo", "command"]},
        lambda repo, command, extra_args=None: _run_project_command(repo, command, extra_args))

    reg.register(
        "cve_lookup",
        "Look up CVEs from the live NVD database — pass an exact CVE ID (CVE-YYYY-NNNNN) or "
        "a keyword (e.g. 'log4j', 'openssl heap overflow'). Free, no API key required.",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "max_results": {"type": "integer", "description": "default 5, max 10"}},
         "required": ["query"]},
        lambda query, max_results=5: _cve_lookup(query, max_results))

    # -- web search ---------------------------------------------------------------
    reg.register(
        "web_search",
        "Search the live web (Tavily). Use for current events, facts not in training data, "
        "or researching a company/person before outreach.",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "max_results": {"type": "integer", "description": "default 5, max 10"}},
         "required": ["query"]},
        lambda query, max_results=5: _web_search(query, max_results))

    # -- job/internship applications -----------------------------------------------
    reg.register(
        "track_application",
        "Log a job/internship application for follow-up tracking. Call this after sending or "
        "drafting outreach to a company.",
        {"type": "object",
         "properties": {"company": {"type": "string"}, "role": {"type": "string"},
                        "contact_email": {"type": "string"}, "notes": {"type": "string"}},
         "required": ["company"]},
        lambda company, role="", contact_email="", notes="":
            f"logged application #{memory.add_application(company, role, contact_email, notes)}")

    reg.register(
        "list_applications",
        "List tracked applications, optionally filtered by status (applied/interview/offer/rejected).",
        {"type": "object", "properties": {"status": {"type": "string"}}},
        lambda status="": memory.list_applications(status) or "no tracked applications")

    reg.register(
        "update_application_status",
        "Update a tracked application's status (e.g. 'interview', 'offer', 'rejected').",
        {"type": "object",
         "properties": {"application_id": {"type": "integer"}, "status": {"type": "string"}},
         "required": ["application_id", "status"]},
        lambda application_id, status: "updated" if memory.update_application_status(application_id, status)
            else "error: no such application")

    # -- gmail + calendar (OAuth; see osiris/google_services.py for setup) --------
    from . import google_services as g

    reg.register(
        "gmail_search", "Search Gmail (standard query: 'is:unread', 'from:foo@bar.com').",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "max_results": {"type": "integer", "description": "default 5"}},
         "required": ["query"]},
        lambda query, max_results=5: g.gmail_search(query, max_results) or "no matching emails")

    reg.register(
        "gmail_read", "Read an email body by message id (from gmail_search).",
        {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]},
        lambda message_id: g.gmail_read(message_id))

    reg.register(
        "gmail_send", "Send an email. Confirm with user before calling.",
        {"type": "object",
         "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "subject", "body"]},
        lambda to, subject, body: g.gmail_send(to, subject, body))

    reg.register(
        "gmail_create_draft",
        "Save an email draft (nothing sends). Call once per email.",
        {"type": "object",
         "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "subject", "body"]},
        lambda to, subject, body: g.gmail_create_draft(to, subject, body))

    reg.register(
        "gmail_reply",
        "Draft a reply to an existing email thread. Saves to Drafts — nothing sends.",
        {"type": "object",
         "properties": {"message_id": {"type": "string", "description": "id of the email to reply to (from gmail_search)"},
                        "body": {"type": "string"}},
         "required": ["message_id", "body"]},
        lambda message_id, body: g.gmail_reply(message_id, body))

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


def _read_pdf(path: str, max_pages: int = 20, max_chars: int = 12000) -> str:
    p = _allowed_path(path)  # same Desktop sandbox as fs_read
    if p.suffix.lower() != ".pdf":
        return f"error: not a PDF: {path} (use fs_read for text files)"
    try:
        import pdfplumber
    except ImportError:
        return "error: pdfplumber not installed — run: pip install pdfplumber"
    try:
        with pdfplumber.open(str(p)) as pdf:
            pages = pdf.pages[:max_pages]
            parts = [(pg.extract_text() or "") for pg in pages]
            more = len(pdf.pages) > max_pages
    except Exception as e:
        return f"error: could not read PDF: {type(e).__name__}: {e}"
    text = "\n\n".join(t for t in parts if t.strip())
    if not text.strip():
        return "(no extractable text — the PDF is likely scanned images, not selectable text)"
    out = text[:max_chars]
    if len(text) > max_chars or more:
        out += "\n…(truncated)"
    return out


def _journal_path(date: str = ""):
    from datetime import date as _date
    day = date or _date.today().isoformat()
    return config.WORKSPACE / "journal" / f"{day}.md", day


def _note(text: str) -> str:
    from datetime import datetime
    p, day = _journal_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(f"# Journal — {day}\n\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%H:%M} — {text}\n")
    return f"noted in journal/{day}.md"


def _read_journal(date: str = "") -> str:
    p, day = _journal_path(date)
    if not p.exists():
        return f"no journal entries for {day}"
    return p.read_text(encoding="utf-8")


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


def _fs_write(path: str, content: str) -> str:
    p = _allowed_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p}"


def _fs_edit(path: str, find: str, replace: str) -> str:
    p = _allowed_path(path)
    text = p.read_text(encoding="utf-8")
    n = text.count(find)
    if n == 0:
        return "error: exact text not found in file — read the file and retry with a verbatim snippet"
    p.write_text(text.replace(find, replace), encoding="utf-8")
    return f"replaced {n} occurrence(s) in {p.name}"


def _scaffold_project(folder: str, files: list) -> str:
    if not isinstance(files, list) or not files:
        return "error: files must be a non-empty array of {path, content}"
    root = _allowed_path(folder)
    written = []
    for f in files:
        if not isinstance(f, dict) or "path" not in f or "content" not in f:
            return f"error: bad file entry {str(f)[:80]} — need {{path, content}}"
        p = (root / f["path"]).resolve()
        if not p.is_relative_to(root):  # no ../ escapes out of the project folder
            return f"error: path escapes project folder: {f['path']}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"], encoding="utf-8")
        written.append(f["path"])
    return f"scaffolded {len(written)} files in {root}:\n" + "\n".join(written)


def _git_diff(repo: str) -> str:
    out = _git(repo, "diff", "--stat", "HEAD")
    if "unknown revision" in out or "bad default revision" in out or "fatal" in out.lower():
        out = _git(repo, "diff", "--stat", "--cached")
    return out[:6000]


def _git_push(repo: str) -> str:
    """Push using an inline gh token so GCM doesn't hang on Windows."""
    import re
    p = _allowed_path(repo)
    remote_url = subprocess.run(["git", "-C", str(p), "remote", "get-url", "origin"],
                                capture_output=True, text=True, timeout=15).stdout.strip()
    if not remote_url:
        return "error: no remote 'origin' set — run github_create_repo first"
    branch = subprocess.run(["git", "-C", str(p), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, timeout=15).stdout.strip() or "main"
    if "github.com" in remote_url:
        token = subprocess.run(["gh", "auth", "token"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        if token:
            clean = re.sub(r"https://[^@]*@", "https://", remote_url)
            authed = clean.replace("https://", f"https://{token}@")
            r = subprocess.run(["git", "-C", str(p), "push", "-u", authed, branch],
                               capture_output=True, text=True, timeout=120)
            return (r.stdout + r.stderr).replace(token, "***").strip() or (
                "pushed" if r.returncode == 0 else f"git push exited {r.returncode}")
    return _git(repo, "push")


def _target_authorized(target: str) -> bool:
    authorized = {t.strip().lower() for t in config.AUTHORIZED_TARGETS.split(",") if t.strip()}
    if target.lower() in authorized:
        return True
    try:
        ip = ipaddress.ip_address(target)
        return ip.is_loopback or ip.is_private
    except ValueError:
        pass
    try:  # hostname that resolves to a private/loopback address counts as local
        ip = ipaddress.ip_address(socket.gethostbyname(target))
        return ip.is_loopback or ip.is_private
    except (socket.gaierror, ValueError):
        return False


def _flag_ok(spec: dict, flag: str) -> bool:
    if flag.startswith("-") or flag.startswith("+"):
        return flag in spec["safe_flags"]
    return bool(re.fullmatch(r"[A-Za-z0-9,.\-]+", flag))  # bare values: ports, record types


def _run_recon(tool: str, target: str, flags: list | None = None) -> str:
    flags = flags or []
    spec = _RECON_TOOLS.get(tool)
    if not spec:
        return f"error: '{tool}' not in allowlist ({', '.join(_RECON_TOOLS)})"
    if not _target_authorized(target):
        return (f"error: '{target}' is not localhost/private and not in OSIRIS_AUTHORIZED_TARGETS "
                f"— add it to .env yourself (comma-separated) only if you have explicit authorization")
    bad = [f for f in flags if not _flag_ok(spec, f)]
    if bad:
        return f"error: flag(s) not allowed for {tool}: {bad} — allowed: {sorted(spec['safe_flags'])}"
    try:
        r = subprocess.run([tool] + flags + [target], capture_output=True, text=True, timeout=90)
    except FileNotFoundError:
        return f"error: '{tool}' is not installed / not on PATH"
    except subprocess.TimeoutExpired:
        return f"error: {tool} timed out after 90s"
    out = (r.stdout + r.stderr).strip()
    return (out[:6000] + ("\n...(truncated)" if len(out) > 6000 else "")) if out else "(no output)"


def _run_project_command(repo: str, command: str, extra_args: list | None = None) -> str:
    extra_args = extra_args or []
    base = _PROJECT_COMMANDS.get(command)
    if not base:
        return f"error: '{command}' not in allowlist ({', '.join(_PROJECT_COMMANDS)})"
    bad = [a for a in extra_args if not _ARG_PATTERN.fullmatch(a)]
    if bad:
        return f"error: extra_args contains disallowed characters: {bad}"
    p = _allowed_path(repo)
    if not p.is_dir():
        return f"error: not a folder: {repo}"
    try:
        r = subprocess.run(base + extra_args, capture_output=True, text=True, timeout=300, cwd=str(p))
    except FileNotFoundError:
        return f"error: '{base[0]}' is not installed / not on PATH"
    except subprocess.TimeoutExpired:
        return f"error: {command} timed out after 300s"
    out = (r.stdout + r.stderr).strip()
    out = (out[:8000] + ("\n...(truncated)" if len(out) > 8000 else "")) if out else "(no output)"
    return f"exit code {r.returncode}\n{out}"


def _cve_lookup(query: str, max_results: int = 5) -> list | str:
    max_results = max(1, min(max_results, 10))
    params = {"resultsPerPage": max_results}
    if re.fullmatch(r"CVE-\d{4}-\d{4,7}", query.upper()):
        params["cveId"] = query.upper()
    else:
        params["keywordSearch"] = query
    try:
        r = requests.get("https://services.nvd.nist.gov/rest/json/cves/2.0",
                         params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        return f"error: NVD lookup failed: {e}"
    out = []
    for item in r.json().get("vulnerabilities", []):
        cve = item["cve"]
        desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
        score = None
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in cve.get("metrics", {}):
                score = cve["metrics"][key][0]["cvssData"].get("baseScore")
                break
        out.append({"id": cve["id"], "published": cve.get("published", "")[:10],
                    "cvss": score, "description": desc[:400]})
    return out or "no matching CVEs"


def _web_search(query: str, max_results: int = 5) -> list | str:
    if not config.TAVILY_API_KEY:
        return "error: TAVILY_API_KEY not set in .env — free key, no card, at tavily.com"
    try:
        r = requests.post("https://api.tavily.com/search",
                          json={"api_key": config.TAVILY_API_KEY, "query": query,
                                "max_results": max(1, min(max_results, 10))},
                          timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        return f"error: web search failed: {e}"
    results = r.json().get("results", [])
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": (x.get("content", "") or "")[:400]} for x in results] or "no results"


def _gh_create_pr(repo: str, title: str, body: str = "", base: str = "main") -> str:
    p = _allowed_path(repo)
    r = subprocess.run(
        ["gh", "pr", "create", "--title", title, "--body", body or "", "--base", base],
        capture_output=True, text=True, timeout=60, cwd=str(p))
    return (r.stdout + r.stderr).strip() or "error: gh pr create returned no output"


def _gh_create_repo(repo: str, name: str = "", private: bool = True) -> str:
    """Create the remote via gh CLI (its keyring auth has repo scope), then push
    with an inline token — the Windows credential manager is known to hang."""
    p = _allowed_path(repo)
    if not (p / ".git").exists():
        return f"error: {p} is not a git repository — git_init and git_commit first"
    name = name or p.name.replace(" ", "-")
    vis = "--private" if private else "--public"
    r = subprocess.run(["gh", "repo", "create", name, vis], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 and "already exists" not in (r.stdout + r.stderr):
        return f"error: gh repo create failed: {(r.stdout + r.stderr).strip()[:300]}"
    login = subprocess.run(["gh", "api", "user", "-q", ".login"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
    token = subprocess.run(["gh", "auth", "token"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
    if not login or not token:
        return "error: gh CLI not authenticated — run 'gh auth login' once"
    subprocess.run(["git", "-C", str(p), "remote", "remove", "origin"], capture_output=True, timeout=15)
    subprocess.run(["git", "-C", str(p), "remote", "add", "origin",
                    f"https://github.com/{login}/{name}.git"], capture_output=True, timeout=15)
    branch = subprocess.run(["git", "-C", str(p), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, timeout=15).stdout.strip() or "main"
    push = subprocess.run(["git", "-C", str(p), "push", "-u",
                           f"https://{token}@github.com/{login}/{name}.git", branch],
                          capture_output=True, text=True, timeout=120)
    if push.returncode != 0:
        return f"error: push failed: {(push.stdout + push.stderr).replace(token, '***')[:300]}"
    return f"created and pushed: https://github.com/{login}/{name} ({'private' if private else 'public'})"


def _git_list_repos() -> list[dict] | str:
    repos = []
    for root in config.ALLOWED_DIRS:
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / ".git").exists():
                    r = subprocess.run(["git", "-C", str(child), "status", "--porcelain", "-b"],
                                       capture_output=True, text=True, timeout=15)
                    lines = r.stdout.splitlines()
                    repos.append({
                        "path": str(child),
                        "branch": lines[0].removeprefix("## ") if lines else "?",
                        "uncommitted_changes": max(0, len(lines) - 1),
                    })
        except (OSError, subprocess.TimeoutExpired):
            continue
    return repos or "no git repositories found in the allowed folders"


_DEFAULT_GITIGNORE = """\
.env
*.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/
node_modules/
dist/
build/
*.egg-info/
venv/
.venv/
*.log
.DS_Store
token.json
credentials.json
"""


def _git_init(path: str) -> str:
    p = _allowed_path(path)
    if (p / ".git").exists():
        return f"error: {p} is already a git repository"
    p.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "-C", str(p), "init", "-b", "main"],
                       capture_output=True, text=True, timeout=30)
    gi = p / ".gitignore"
    if not gi.exists():
        gi.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
    return (r.stdout + r.stderr).strip() or f"initialized empty repo at {p}"


def _gh_list_repos() -> list[dict]:
    r = requests.get("https://api.github.com/user/repos", headers=_gh_headers(),
                     params={"per_page": 20, "sort": "pushed"}, timeout=15)
    r.raise_for_status()
    return [{"name": x["full_name"], "private": x["private"],
             "last_push": (x.get("pushed_at") or "")[:10]} for x in r.json()]


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
