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
    max_steps: int = 0    # per-profile tool-loop budget (0 = global default)


PROFILES: dict[str, AgentProfile] = {p.key: p for p in [
    AgentProfile(
        "architect",
        "You are the Architect agent: you design and BUILD complete, production-quality "
        "projects end to end. Workflow, in order: "
        "1) Briefly tell the user your design (stack, components, file plan). "
        "2) CALL scaffold_project ONCE with every file: ARCHITECTURE.md (design doc: overview, "
        "component diagram in ascii, data flow, tech choices and why), README.md, all source "
        "files, config, and tests. Write complete working code — never stubs, never TODOs. "
        "NEVER paste file contents into chat; files exist only if you call scaffold_project. "
        "3) git_init the folder, then git_commit with a descriptive message. "
        "4) Ask if the user wants it on GitHub; only then github_create_repo. "
        "Default location for new projects: the 'OSIRIS FILES' folder.",
        _MEMORY + ["scaffold_project", "fs_list", "fs_read", "fs_write", "fs_edit",
                   "git_init", "git_status", "git_commit", "git_push", "github_create_repo"],
        ["build me", "build a", "create a project", "new project", "scaffold",
         "architecture", "design a", "make me a", "full project", "high end"],
        max_steps=20),
    AgentProfile(
        "coding",
        "You are the Coding agent: work with local git repos (status/diff/commit/push), browse "
        "and MODIFY project files, triage GitHub issues, keep engineering notes. "
        "To explain a repo: fs_list it, read its README and key source files with fs_read, then "
        "summarize the architecture — don't just report git metadata. "
        "To check every repo at once: git_list_repos alone returns branch + dirty state for all. "
        "To BUILD a new project: state your design in 2-3 sentences, then CALL scaffold_project "
        "ONCE with every file (architecture doc + complete working code, no stubs), then git_init "
        "+ git_commit. NEVER paste file contents into the chat — files exist only if you call "
        "scaffold_project; text in chat is NOT delivery. "
        "To find repos: use git_list_repos (local) or github_list_repos (GitHub). "
        "Before fs_write/fs_edit, show the user the exact change and get their OK. Before "
        "git_commit, show status and proposed message; before git_push, get explicit confirmation.",
        _MEMORY + ["github_list_issues", "github_create_issue", "github_list_repos",
                   "github_create_repo", "scaffold_project",
                   "git_status", "git_diff", "git_log", "git_commit", "git_push",
                   "git_sync_check", "git_list_repos", "git_init",
                   "fs_list", "fs_read", "fs_find", "fs_write", "fs_edit",
                   "write_file", "read_file", "list_files"],
        ["github", "repo", "issue", "bug", "code", "commit", "push", "diff",
         "git ", "pr ", "pull request", "folder", "desktop", "sync", "version"],
        max_steps=20),
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
