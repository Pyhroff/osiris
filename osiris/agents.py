"""Specialized agents: each is a persona + a tool subset.

Keeping the tool list per-agent small matters — fewer tools in context means
cheaper calls and sharper tool selection, especially on smaller models.
Routing is keyword-based (free, instant); ambiguous input falls through to
the personal assistant, which carries every tool.
"""
from __future__ import annotations

from dataclasses import dataclass

_MEMORY = ["remember", "recall", "current_datetime"]


@dataclass(frozen=True)
class AgentProfile:
    key: str
    persona: str          # appended to the base system prompt
    tools: list[str]      # tool names this agent may see (empty = all)
    keywords: list[str]   # routing triggers


PROFILES: dict[str, AgentProfile] = {p.key: p for p in [
    AgentProfile(
        "coding",
        "You are the Coding agent: work with local git repos (status/diff/commit/push), browse "
        "and MODIFY project files, triage GitHub issues, keep engineering notes. "
        "To explain a repo: fs_list it, read its README and key source files with fs_read, then "
        "summarize the architecture — don't just report git metadata. "
        "To check every repo at once: git_list_repos alone returns branch + dirty state for all. "
        "To create a new repo: git_init, then fs_write files, then git_commit. "
        "To find repos: use git_list_repos (local) or github_list_repos (GitHub). "
        "Before fs_write/fs_edit, show the user the exact change and get their OK. Before "
        "git_commit, show status and proposed message; before git_push, get explicit confirmation.",
        _MEMORY + ["github_list_issues", "github_create_issue", "github_list_repos",
                   "git_status", "git_diff", "git_log", "git_commit", "git_push",
                   "git_sync_check", "git_list_repos", "git_init",
                   "fs_list", "fs_read", "fs_find", "fs_write", "fs_edit",
                   "write_file", "read_file", "list_files"],
        ["github", "repo", "issue", "bug", "code", "commit", "push", "diff",
         "git ", "pr ", "pull request", "folder", "desktop", "sync", "version"]),
    AgentProfile(
        "email",
        "You are the Email agent: search, read, summarize, and draft mail. Prefer "
        "gmail_create_draft so the user reviews before sending; for several emails, create "
        "one draft each. Never gmail_send without explicit confirmation in this conversation.",
        _MEMORY + ["gmail_search", "gmail_read", "gmail_send", "gmail_create_draft"],
        ["email", "mail", "inbox", "unread", "gmail", "draft", "send a message to"]),
    AgentProfile(
        "learning",
        "You are the Learning agent: plan long-horizon goals, track milestones, schedule study.",
        _MEMORY + ["create_plan", "list_goals", "complete_milestone",
                   "schedule_task", "list_tasks", "complete_task"],
        ["learn", "study", "goal", "plan ", "milestone", "course", "roadmap"]),
    AgentProfile(
        "research",
        "You are the Research agent: gather what is known from Drive and memory, synthesize, "
        "and save findings as organized notes in workspace files.",
        _MEMORY + ["write_file", "read_file", "list_files", "drive_search", "drive_read",
                   "fs_list", "fs_read", "fs_find"],
        ["research", "investigate", "summarize", "compare", "analyze", "notes on",
         "drive", "document", "docs"]),
    AgentProfile(
        "personal",
        "You are the Personal Assistant: the default agent with access to everything.",
        [],  # empty = full registry
        []),
]}


def pick_agent(text: str) -> AgentProfile:
    t = text.lower()
    best, hits = PROFILES["personal"], 0
    for p in PROFILES.values():
        n = sum(1 for k in p.keywords if k in t)
        if n > hits:
            best, hits = p, n
    return best
