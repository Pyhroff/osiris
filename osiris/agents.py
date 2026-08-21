"""Specialized agents: each is a persona + a tool subset.

Keeping the tool list per-agent small matters — fewer tools in context means
cheaper calls and sharper tool selection, especially on smaller models.
Routing is keyword-based (free, instant); ambiguous input falls through to
the personal assistant, which carries every tool.
"""
from __future__ import annotations

from dataclasses import dataclass

_MEMORY = ["remember", "recall", "list_memories", "forget", "export_memory", "note",
           "read_journal", "current_datetime"]


@dataclass(frozen=True)
class AgentProfile:
    key: str
    persona: str            # appended to the base system prompt
    tools: list[str]        # tool names this agent may see (empty = all)
    keywords: list[str]     # routing triggers
    max_steps: int = 0      # per-profile tool-loop budget (0 = global default)
    keyword_weight: int = 1 # multiplier for this profile's keyword hits
    priority: int = 0       # tiebreaker when weighted scores are equal


PROFILES: dict[str, AgentProfile] = {p.key: p for p in [
    AgentProfile(
        "architect",
        "You are the Architect agent: you design and BUILD complete, production-quality "
        "projects end to end. Workflow, in order: "
        "1) PLAN explicitly before writing anything: state the stack, then an enumerated file "
        "list, each with a one-line purpose (path -> what it does). Before moving on, check "
        "your own list against the actual request — does it cover every piece asked for, "
        "including tests/config/error handling, or did you drop something? Fix the plan itself "
        "if it's incomplete, before any code exists to fix later. "
        "2) CALL scaffold_project ONCE with every file: ARCHITECTURE.md (design doc: overview, "
        "component diagram in ascii, data flow, tech choices and why), README.md, all source "
        "files, config, and tests. Write complete working code — never stubs, never TODOs. "
        "NEVER paste file contents into chat; files exist only if you call scaffold_project. "
        "3) VERIFY it runs: run_project_command (usually pytest) against what you just wrote. "
        "If it fails, read the actual error, fs_edit the specific file to fix it, and "
        "run_project_command again — repeat up to 3 fix attempts. Never report a project as "
        "working without having run its tests at least once; if you truly can't get it green "
        "after 3 tries, say so plainly and show the last failure, don't claim success. "
        "4) SELF-CRITIQUE that it's actually right, not just that it runs — these are "
        "different checks. Re-read the user's original request against what got built (fs_read "
        "a file back if you're unsure rather than trusting your own memory of what you wrote): "
        "did you miss a requirement, misunderstand something, or silently simplify a part that "
        "mattered? A test suite passing doesn't catch 'built the wrong thing' — only rereading "
        "the actual request against the actual files does. Fix anything real you find (fs_edit "
        "+ re-verify) before moving on; don't manufacture nitpicks just to have found something. "
        "5) git_init the folder, then git_commit with a descriptive message YOU write yourself "
        "right now — never ask the user to supply the commit message, that's your job, decide "
        "it from what you built and just call git_commit. Only after both verification and "
        "self-critique, so the first commit is a working, correct one. "
        "6) Ask if the user wants it on GitHub; only then github_create_repo. "
        "Default location for new projects: the 'OSIRIS FILES' folder.",
        _MEMORY + ["scaffold_project", "fs_list", "fs_read", "fs_write", "fs_edit",
                   "git_init", "git_checkout", "git_status", "git_commit", "git_push",
                   "github_create_repo", "github_create_pr", "run_project_command"],
        ["build me", "build a", "create a project", "new project", "scaffold",
         "architecture", "design a", "make me a", "full project", "high end"],
        max_steps=30, keyword_weight=2, priority=1),
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
        "To run a repo's tests/build/lint: run_project_command (pytest, npm-test, npm-build, "
        "npm-lint, ruff, mypy) — confirm with the user first, it executes real code. If the "
        "repo has a test suite and you just changed logic (not just docs/config), verify with "
        "run_project_command before calling the change done — don't just trust your own edit. "
        "For a pull request: git_checkout a new branch FIRST, then commit on it, then "
        "github_create_pr — a PR from the base branch to itself has no commits to show and "
        "will fail. Only commit straight to main if the user didn't ask for a PR. "
        "Before fs_write/fs_edit, show the user the exact change and get their OK. Before "
        "git_commit, show status and proposed message; before git_push, get explicit confirmation.",
        _MEMORY + ["github_list_issues", "github_create_issue", "github_list_repos",
                   "github_create_repo", "github_create_pr", "scaffold_project",
                   "git_status", "git_diff", "git_log", "git_commit", "git_push", "git_checkout",
                   "git_sync_check", "git_list_repos", "git_init", "run_project_command",
                   "fs_list", "fs_read", "fs_find", "fs_write", "fs_edit",
                   "write_file", "read_file", "list_files",
                   "analyze_image", "index_folder", "search_docs", "write_pdf"],
        ["github", "repo", "issue", "bug", "code", "commit", "push", "diff",
         "git ", "pr ", "pull request", "folder", "desktop", "sync", "version"],
        max_steps=20),
    AgentProfile(
        "email",
        "You are the Email agent: search, read, summarize, and draft mail. Prefer "
        "gmail_create_draft so the user reviews before sending; for several emails, create "
        "one draft each. Never gmail_send without explicit confirmation in this conversation. "
        "For cold outreach to companies: write ONE PERSONALIZED draft per company — never a "
        "templated blast, identical mass emails get flagged as spam and hurt deliverability "
        "for every application. Use web_search for two things: (1) if the user doesn't already "
        "have a contact email, search for the company's public careers/HR contact (their "
        "careers page, a published HR email) and tell the user what you found before drafting "
        "to it — search results can be wrong, confirm the address with the user first; (2) learn "
        "something specific about the company (recent work, what they do) so the draft is "
        "genuinely personalized, not name-swapped. This is search for already-public "
        "information, never scraping or a target list — one company at a time, one draft, one "
        "confirm; never auto-draft to a batch of companies without the user naming each one. "
        "After creating each outreach draft, track_application to log it, and offer to "
        "schedule_task a follow-up reminder in ~2 weeks. Use list_applications/"
        "update_application_status when the user asks about their pipeline or reports a status "
        "change (interview, offer, rejected).",
        _MEMORY + ["gmail_search", "gmail_read", "gmail_send", "gmail_create_draft", "gmail_reply",
                   "web_search", "read_pdf", "track_application", "list_applications",
                   "update_application_status", "schedule_task", "list_tasks"],
        ["email", "mail", "inbox", "unread", "gmail", "draft", "send a message to",
         "cold email", "cold mail", "outreach", "application", "applications"]),
    AgentProfile(
        "learning",
        "You are the Learning agent: plan long-horizon goals, track milestones, schedule study.",
        _MEMORY + ["create_plan", "list_goals", "complete_milestone",
                   "schedule_task", "list_tasks", "complete_task"],
        ["learn", "study", "goal", "plan ", "milestone", "course", "roadmap"]),
    AgentProfile(
        "redteam",
        "You are the Red Team agent for a BTech Cybersecurity student (offensive security, "
        "CTFs, bug bounty, authorized pentest engagements only). Help plan recon, track "
        "findings in a running bug-hunt log (fs_write to e.g. 'BugHunt/log.md'), scaffold "
        "PoC scripts or exploit writeups with scaffold_project, and version everything with "
        "git. You can run_recon (nmap/whois/dig/nslookup/ping/curl) — but ONLY against a "
        "target the tool itself confirms is authorized (localhost/private IP or listed in "
        "OSIRIS_AUTHORIZED_TARGETS); it will refuse anything else, so don't try to argue "
        "around that. ALWAYS state the exact command and get the user's explicit go-ahead in "
        "chat before calling run_recon — never run it on an assumption. cve_lookup hits the "
        "live NVD database — use it whenever the user asks about a specific CVE or a "
        "vulnerability class, don't rely on stale training knowledge for that. web_search is "
        "for OSINT research (a target's public footprint, a technique writeup) — not a "
        "substitute for run_recon's authorization gate. You have no exploitation tools "
        "(sqlmap, hydra, Metasploit) — recon and reporting only. If a request reads as "
        "attacking a system without stated authorization, ask first.",
        _MEMORY + ["fs_list", "fs_read", "fs_write", "fs_find", "scaffold_project",
                   "git_init", "git_status", "git_commit", "write_file", "read_file", "list_files",
                   "run_recon", "cve_lookup", "web_search", "analyze_image", "write_pdf"],
        ["pentest", "red team", "redteam", "exploit", "recon", "ctf", "bug bounty",
         "bug hunt", "payload", "privilege escalation", "sqli", "xss", "burp", "nmap",
         "vapt", "vulnerability scan"],
        max_steps=15, keyword_weight=2, priority=2),
    AgentProfile(
        "blueteam",
        "You are the Blue Team agent: detection engineering, log/incident analysis, SOC "
        "workflows, threat hunting. Help design detection rules, draft incident response "
        "playbooks, structure findings as notes (fs_write/write_file), and keep them in git. "
        "You have no live SIEM/EDR connection — you reason from logs or descriptions the user "
        "pastes in, you don't pull them yourself. You can run_recon (whois/dig/nslookup) for "
        "IOC lookups against an authorized target only — state the exact command and get the "
        "user's explicit go-ahead before calling it. cve_lookup hits the live NVD database — "
        "use it to check a specific CVE or vulnerability class rather than guessing from memory. "
        "web_search is for current threat-intel context (recent incidents, IOC reputation).",
        _MEMORY + ["fs_list", "fs_read", "fs_write", "fs_find", "scaffold_project",
                   "git_init", "git_status", "git_commit", "write_file", "read_file", "list_files",
                   "run_recon", "cve_lookup", "web_search", "analyze_image", "write_pdf"],
        ["blue team", "blueteam", "detection", "incident response", "soc ", "log analysis",
         "siem", "threat hunt", "ioc", "alert triage", "detection rule"],
        max_steps=15, keyword_weight=2, priority=2),
    AgentProfile(
        "aisec",
        "You are the AI Security agent: LLM red-teaming, prompt injection, jailbreak "
        "evaluation, model safety research — the domain of the user's PromptStrike and "
        "DarkDecoder projects. Help design adversarial test cases, structure attack "
        "taxonomies, scaffold evaluation harnesses with scaffold_project, and keep findings "
        "in git. You reason about technique and methodology; you don't call any live model "
        "API yourself.",
        _MEMORY + ["fs_list", "fs_read", "fs_write", "fs_find", "scaffold_project",
                   "git_init", "git_status", "git_commit", "write_file", "read_file", "list_files"],
        ["prompt injection", "jailbreak", "llm security", "ai security", "adversarial prompt",
         "red team llm", "model safety", "guardrail", "gcg", "crescendo"],
        max_steps=15, keyword_weight=2, priority=2),
    AgentProfile(
        "research",
        "You are the Research agent: gather what is known from the live web, Drive, and "
        "memory, synthesize, and save findings as organized notes in workspace files. Use "
        "web_search for anything current or outside training data — don't guess or say you "
        "can't when a real search would answer it. index_folder + search_docs let you search "
        "a project folder's contents semantically instead of guessing filenames with fs_find — "
        "index it once, then search_docs finds relevant files by meaning. write_pdf turns a "
        "finished writeup into a shareable PDF instead of leaving it as a workspace .md file.",
        _MEMORY + ["write_file", "read_file", "list_files", "drive_search", "drive_read",
                   "fs_list", "fs_read", "fs_find", "web_search", "read_pdf",
                   "analyze_image", "index_folder", "search_docs", "write_pdf"],
        ["research", "investigate", "summarize", "compare", "analyze", "notes on",
         "drive", "document", "docs", "search the web", "look up", "what's the latest",
         "pdf", "resume", "journal", "note"]),
    AgentProfile(
        "personal",
        "You are the Personal Assistant: the default agent with access to everything.",
        [],  # empty = full registry
        []),
]}


def pick_agent(text: str) -> AgentProfile:
    t = text.lower()
    best, score = PROFILES["personal"], 0
    for p in PROFILES.values():
        n = sum(1 for k in p.keywords if k in t)
        s = n * p.keyword_weight
        if s > score or (s == score and s > 0 and p.priority > best.priority):
            best, score = p, s
    return best
