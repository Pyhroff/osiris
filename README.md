# OSIRIS

![tests](https://github.com/Pyhroff/osiris/actions/workflows/ci.yml/badge.svg)

Personal AI operating system — intelligence as the kernel. Model-agnostic,
multi-agent, with persistent memory, long-horizon planning, a task scheduler,
sandboxed files, and Gmail/Calendar/Drive/GitHub/Discord integration. Runs
entirely on **free** API tiers (Gemini, Groq, Cerebras, Mistral, OpenRouter,
GitHub Models, local Ollama).

```
OSIRIS/
├── Core          planner · memory (SQLite+embeddings) · scheduler · model router
├── Agents        9 specialized personas, keyword-routed, per-agent tool subsets
├── Integrations  local git · desktop files · GitHub · Gmail · Calendar · Drive · Discord
├── Providers     Gemini → Groq → Cerebras → Mistral → OpenRouter → GitHub Models → Ollama → OpenAI
└── Frontend      CRT-terminal web chat + live goals/tasks panel
```

**Model Router** — OSIRIS never depends on one provider. It tries them in
priority order and fails over on any error (rate limit, outage, bad key),
cools down whichever failed, and retries the full chain once before giving up:

```
                                    OSIRIS
                                       │
                                Model Router
                                       │
 ┌────────┬────────┬──────────┬─────────┬────────────┬──────────────┬────────┬────────┐
 Gemini    Groq     Cerebras   Mistral   OpenRouter    GitHub Models  Ollama   OpenAI
 Flash     free      free      free      :free pool    gpt-4o-mini    local    (optional)
```

Any provider without a key (or, for Ollama, not running) is skipped for free —
list them all in `OSIRIS_PROVIDERS` and only the reachable ones are used.

**9 specialized agents** — each request is keyword-routed to the agent that
fits, and each agent only sees its own tool subset (cheaper calls, sharper
tool choice). Ambiguous input falls through to `personal`, which carries
every tool:

| Agent | Focus | Key tools |
|---|---|---|
| architect | Plans a project, scaffolds it whole, runs its tests, self-critiques against the original ask, commits, offers GitHub | scaffold_project, run_project_command, git_*, github_create_repo/pr |
| coding | Local git workflow, reads/explains/edits repos, GitHub issues & PRs | git_*, github_*, fs_*, scaffold_project |
| email | Gmail search/read/draft/reply, personalized cold outreach, application tracking | gmail_*, track_application, web_search |
| learning | Long-horizon goal → milestone planning, scheduling | create_plan, list_goals, schedule_task |
| redteam | Recon, CVE lookups, bug-hunt logs, PoC scaffolding — authorized targets only, no exploitation tools | run_recon, cve_lookup, scaffold_project |
| blueteam | Detection engineering, IR playbooks, IOC lookups, threat-intel search | run_recon, cve_lookup, web_search |
| aisec | LLM red-teaming / jailbreak methodology, adversarial test design | fs_*, scaffold_project, git_* |
| research | Web/Drive/semantic-doc search, synthesis into saved notes or a PDF | web_search, drive_*, index_folder, search_docs, write_pdf |
| personal | Default fallback — every tool, incl. Discord + Calendar | (all ~40) |

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env        # then paste your key(s)

python server.py              # web UI → http://localhost:8765
# or
python main.py                # terminal REPL
```

### `osiris` from any terminal (optional)

```bash
copy osiris.cmd "%LOCALAPPDATA%\Microsoft\WindowsApps\"
# edit the copied file: set OSIRIS_HOME to your clone's absolute path
```

Then from anywhere: `osiris` (REPL) · `osiris serve` (web UI) ·
`osiris daemon` (reminders) · `osiris bot` (Discord mobile dispatcher).

You need **at least one** provider (all free options):
- Gemini: https://aistudio.google.com/apikey (recommended — also powers memory embeddings and vision)
- Groq: https://console.groq.com/keys
- Cerebras: https://cloud.cerebras.ai (fastest 70B inference anywhere)
- Mistral: https://console.mistral.ai (La Plateforme → API keys, free tier)
- OpenRouter: https://openrouter.ai/keys (`:free` model pool)
- GitHub Models: a GitHub PAT with `Models: read` → free gpt-4o-mini
- Ollama: just have it running locally — auto-detected, no key
- OpenAI: optional paid fallback

Optional: add a `GITHUB_TOKEN` to enable the GitHub tools, and a `TAVILY_API_KEY`
(free, no card: tavily.com) to enable `web_search`.

### Gmail + Calendar + Drive (optional, one-time)

1. In [console.cloud.google.com](https://console.cloud.google.com) (same project as your
   Gemini key): **APIs & Services → Library** → enable **Gmail API**, **Google Calendar API**,
   and **Google Drive API**
2. **Credentials → Create OAuth client ID → Desktop app** → download the JSON
3. Save it as `credentials.json` in this folder

The first Google request opens your browser to authorize; the token is cached in
`token.json` afterwards. Both files are gitignored. (If you authorized before Drive
was added, delete `token.json` once to pick up the new scope.)

### Discord — two independent integrations

**Webhook (send-only notifications):** Channel settings → **Integrations →
Webhooks → New Webhook** → copy the URL into `DISCORD_WEBHOOK_URL` in `.env`.
Enables the `discord_send` tool, and `daemon.py` mirrors due-task reminders
to that channel.

**Bot (mobile dispatcher, full two-way chat):** create a Discord bot, set
`DISCORD_BOT_TOKEN` and `OSIRIS_DISCORD_ALLOWED_USERS` (your Discord user
id, comma-separated for more than one) in `.env`, then run `python
discord_bot.py` (or `osiris bot`). Only DMs from allowlisted user ids get a
response — everyone else is silently ignored, no hint the bot exists. This
process also polls tasks/routines internally, so it replaces `daemon.py`
when running (one process covers chat + proactive reminders).

### Background daemon (optional, if not running the Discord bot)

```bash
python daemon.py        # in a second terminal
```

Polls `osiris.db` every 60 s and fires a Windows toast (+ Discord webhook, if
set) when a scheduled task or a `routines.json` routine comes due. Reminders
cost zero API quota; routines run the full agent and save a briefing to
`workspace/briefings/`.

## Try it

```
you › remember that I'm targeting IBM Quantum roles after graduating in 2027
you › I want to get into Stanford AI in 5 years          ← it will offer to plan it
you › remind me tomorrow 9am to review my PhantomGrid threat model
you › is PromptStrike in sync with GitHub? if not, commit and push it
you › draft replies to my 3 unread emails — I'll review them in Gmail
you › build me a full project: a CLI tool that tags TODOs across a repo
you › look up CVE-2024-3094
you › index my StylesOS folder, then find where the theme colors are set
you › any unread emails about the hackathon?
you › what's on my calendar this week?
you › what do you know about me?
you › /goals
```

Memory, goals, tasks, and tracked job applications persist in `osiris.db`
across sessions. Due tasks are announced at startup.

### REPL-only commands (no LLM call, instant, free)

```
/status    live-verified health check — actually pings Gmail/GitHub/Discord/Ollama,
           not just "is a key present" (a token can be present and dead)
/backup    export memories/goals/tasks/applications to workspace/backups/ right now
/goals /tasks /model /reflect /clear /help
```

`export_memory` is also a regular tool, so you can just ask: *"back up my memory"*.
Unlike code, nothing else backs up `osiris.db` — worth running `/backup` (or asking
for it) periodically, since a lost/corrupted db has no other copy.

## Tests

```bash
pytest tests/ -q
```

77 tests, fully offline — the LLM is stubbed, git tests run against a
throwaway temp repo, and integration tools are verified to fail gracefully
without credentials.

## Architecture notes

- **Model-agnostic by design.** The agent speaks a neutral message/tool format
  ([osiris/providers/base.py](osiris/providers/base.py)). Groq, Cerebras, Mistral,
  OpenRouter, GitHub Models, Ollama, and OpenAI all share one OpenAI-compatible
  adapter; adding any new provider = one small class.
- **Failover, not fragility.** Gemini rate-limited? The router silently retries the
  same request down the chain. Set priority with `OSIRIS_PROVIDERS` in `.env`.
- **Guarded local access.** `fs_*` (read/list/find/write/edit) and `git_*` tools are
  confined to allow-listed roots (your real Windows Desktop by default, resolved via
  the registry so OneDrive redirection can't silently misplace a file;
  `OSIRIS_ALLOWED_DIRS` to change) — system paths are rejected. `write_file`/`read_file`
  are further sandboxed to `./workspace/`. Agents are instructed to show status/diffs
  before committing and get explicit confirmation before pushing, sending, or running
  recon.
- **Authorization-gated recon.** `run_recon` (nmap/whois/dig/nslookup/ping/curl) only
  runs against localhost, a private IP, or a host explicitly listed in
  `OSIRIS_AUTHORIZED_TARGETS` — anything else is refused outright, with a per-tool
  flag allowlist (no file-writing or script-execution flags). The `redteam`/`blueteam`
  agents have no exploitation tools (sqlmap, hydra, Metasploit) — recon and reporting
  only.
- **New front door, same permissions.** The Discord bot is a new *surface*, not new
  *capability* — every tool's own guardrails apply identically whether the request
  comes from the REPL, the web UI, or a DM.
- **`/status` verifies, it doesn't just check for a file.** A `credentials.json` on
  disk or a `GITHUB_TOKEN` string in `.env` proves nothing about whether it still
  *works* — tokens expire and get revoked silently. `/status` actually refreshes/pings
  Gmail, GitHub, Discord's webhook, and Ollama, so a dead credential surfaces before
  you're mid-conversation with a confused agent, not after.
- **This folder may itself be cloud-synced (OneDrive, etc.).** Being `.gitignore`d
  keeps `osiris.db`/`credentials.json`/`token.json`/`workspace/` off GitHub, but it
  doesn't stop a syncing folder from uploading them elsewhere — OSIRIS prints a
  one-time startup note if it detects this (`OSIRIS_ACK_ONEDRIVE=1` to silence it).
  Not necessarily a problem — it doubles as an off-machine backup — just worth being
  a deliberate choice rather than an invisible one.

## Roadmap

- [x] Gmail + Google Calendar + Google Drive via OAuth (same Google Cloud project as the API key)
- [x] Background daemon: fire due tasks as notifications instead of at-startup
- [x] Model router: 8-provider failover chain (Gemini, Groq, Cerebras, Mistral, OpenRouter, GitHub Models, Ollama, OpenAI)
- [x] 9 specialized agents with per-agent tool subsets, incl. dedicated redteam/blueteam/aisec personas
- [x] Discord: send-only webhook + daemon reminders, and a full two-way mobile bot (DM-allowlisted)
- [x] Web frontend: terminal-style chat with live goals/tasks panel
- [x] Reflection pass: sessions are distilled into long-term memories
      (`/reflect`, on REPL exit, and every 8 web turns in the background —
      deduplicated by embedding similarity)
- [x] CI: pytest on Ubuntu + Windows
- [x] Vision: `analyze_image` — describe or answer questions about a screenshot/photo (Gemini)
- [x] Local document index: `index_folder` + `search_docs` — semantic search over a project
      folder's files, not just fs_find-by-name
- [x] `write_pdf` — generate a PDF report/briefing from scratch, not just workspace `.md` notes
- [x] Architect agent: plan → scaffold → test → self-critique → commit, end-to-end project builds
- [x] Authorized recon (`run_recon`) + live CVE lookup (`cve_lookup`) for the security agents
- [x] Job/internship application tracking (`track_application`, `list_applications`, status updates)
- [x] Memory backup/export (`export_memory` tool, `/backup`) — `osiris.db` has no other copy
- [x] `/status` does live validity checks (refresh/ping), not just "is a key present"
- [x] OneDrive/cloud-sync detection — one-time startup note if the project folder itself syncs
- [ ] Knowledge graph over memories (entities + relations, not just embeddings)
- [ ] Notion integration
