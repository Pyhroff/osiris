# OSIRIS

![tests](https://github.com/Pyhroff/osiris/actions/workflows/ci.yml/badge.svg)

A model-agnostic agent runtime built as a real, daily-used personal AI
assistant — not a demo. The engineering surface: **provider abstraction**
(8 backends, automatic failover, one shared adapter), **agentic
tool-calling** (9 keyword-routed personas, ~40 tools, per-agent scoping,
circuit-breaker retry logic), **RAG-style memory** (embeddings + cosine
recall with graceful keyword fallback, session-to-memory reflection), and
**local-model reliability engineering** (see [Measured, not assumed](#measured-not-assumed)
below — this isn't just prompting, it's provider-quirk debugging with
receipts). Runs entirely on **free** API tiers (Gemini, Groq, Cerebras,
Mistral, OpenRouter, GitHub Models, local Ollama).

One of the 9 agent personas happens to be security-focused (the author's
day job/degree) — see the agent table below — but that's a subset, not
the point of the system.

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

## Measured, not assumed

`_OLLAMA_RANK` (which local model OSIRIS prefers) used to be an educated
guess, ranked by assumed tool-calling competence. It's an educated guess in
a lot of agent frameworks — provider docs claim reliability, nobody actually
measures it against the specific models people have pulled. So I built a
real eval harness ([`benchmarks/`](benchmarks/)) against OSIRIS's own tool
registry and production code path — not a toy schema, not a mock — and ran
it against 3 locally-pulled models. Full methodology and raw results:
[`benchmarks/FINDINGS.md`](benchmarks/FINDINGS.md). The short version:

- **Confirmed, not assumed**: one model (`gemma3`) can't do tool-calling on
  Ollama at all — verified live via the actual API error, not inferred.
- **A real tradeoff, quantified, then fixed at the root**: the most-accurate
  model (`qwen3`, 10/10 correct in testing) took 37–155 seconds per call on
  CPU-only hardware — a "thinking" model burning hundreds of reasoning
  tokens before every answer. Traced the fix (`think: false`) to Ollama's
  *native* API; found it was silently ignored on the OpenAI-compatible
  endpoint OSIRIS called at the time — so Ollama now gets its own adapter
  ([`OllamaNativeLLM`](osiris/providers/ollama_native.py)) instead of
  sharing one with 6 other providers. Verified live: 100s→10s on a plain
  task, ~2.5–3x on a real tool-calling request.
- **A bug found and fixed**: a weaker model (`llama3.2`) consistently
  confused two tools with overlapping-sounding descriptions. Ruled out the
  test harness as the cause first (re-ran against OSIRIS's real system
  prompt, same failure), then fixed it with a tightened tool description,
  verified by re-test. A second, similar-looking confusion did *not*
  respond to the same fix after two attempts — reported as an open
  limitation rather than forced into a clean result.

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
- Cerebras: https://cloud.cerebras.ai (fastest 70B inference anywhere — note: as of
  2026 this is expiring promotional credits, not a permanent standing free tier
  like Gemini/Groq, so it can eventually need a top-up)
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

Send a photo/screenshot in the DM (with or without a caption) and it's saved to
`workspace/inbox/` and handed straight to `analyze_image` — "what's this error"
+ a screenshot from your phone works, not just text. Non-image attachments
aren't handled yet.

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

100 tests, fully offline — the LLM is stubbed, git tests run against a
throwaway temp repo, and integration tools are verified to fail gracefully
without credentials.

## Architecture notes

- **Model-agnostic by design.** The agent speaks a neutral message/tool format
  ([osiris/providers/base.py](osiris/providers/base.py)). Groq, Cerebras, Mistral,
  OpenRouter, GitHub Models, and OpenAI all share one OpenAI-compatible adapter;
  adding any new provider = one small class. Ollama is the one exception — it gets
  its own native-API adapter ([ollama_native.py](osiris/providers/ollama_native.py))
  specifically because its OpenAI-compat endpoint silently ignores `think:false`
  (verified live) — a real, measured reason to break the pattern, not a default.
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
- [x] Discord bot handles image attachments — a phone photo now reaches `analyze_image`, not just text
- [x] Measured local-model tool-calling benchmark (`benchmarks/`) — replaced a guessed model
      ranking with actual data, found and fixed a real tool-description bug
- [x] `OllamaNativeLLM` — Ollama gets its own native-API code path instead of sharing the
      OpenAI-compat adapter, specifically so `think:false` actually takes effect (it's
      silently ignored on the compat endpoint). Verified live: 100s→10s on a plain generation
      task, ~2.5-3x on a real tool-calling request (less than 10x there — tool selection
      itself has generation overhead even with reasoning suppressed). `OSIRIS_OLLAMA_THINK`
      in `.env` to override the default.
- [ ] Knowledge graph over memories (entities + relations, not just embeddings)
- [ ] Notion integration
