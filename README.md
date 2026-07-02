# OSIRIS

![tests](https://github.com/Pyhroff/osiris/actions/workflows/ci.yml/badge.svg)

Personal AI operating system — intelligence as the kernel. Model-agnostic,
multi-agent, with persistent memory, long-horizon planning, a task scheduler,
sandboxed files, and Gmail/Calendar/GitHub integration. Runs entirely on
**free** API tiers (Gemini, Groq, local Ollama).

```
OSIRIS/
├── Core          planner · memory (SQLite+embeddings) · scheduler · model router
├── Agents        coding · email · learning · research · personal assistant
├── Integrations  local git · desktop files · GitHub · Gmail · Calendar · Drive · Discord
├── Providers     Gemini → Groq → Ollama → OpenAI (automatic failover)
└── Frontend      CRT-terminal web chat + live goals/tasks panel
```

**Model Router** — OSIRIS never depends on one provider. It tries them in
priority order and fails over on any error (rate limit, outage, bad key):

```
             OSIRIS
                │
         Model Router
                │
 ┌─────────┬─────────┬─────────┬─────────┐
 Gemini     Groq      Ollama    OpenAI
 Flash      Llama     Qwen      (optional)
```

**Specialized agents** — each request is routed to the agent that fits, and
each agent only sees its own tool subset (cheaper calls, sharper tool choice):

| Agent | Tools |
|---|---|
| coding | local git (status/diff/log/commit/push/sync-check), desktop files, GitHub issues, memory |
| email | Gmail search/read/drafts/send, memory |
| learning | goal planner, milestones, scheduler, memory |
| research | Google Drive search/read, desktop files, workspace notes, memory |
| personal | everything incl. Discord + Calendar (default fallback) |

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

Then from anywhere: `osiris` (REPL) · `osiris serve` (web UI) · `osiris daemon` (reminders).

You need **at least one** provider (all free options):
- Gemini: https://aistudio.google.com/apikey (recommended — also powers memory embeddings)
- Groq: https://console.groq.com/keys
- Ollama: just have it running locally — auto-detected, no key

Optional: add a `GITHUB_TOKEN` to enable the GitHub tools.

### Gmail + Calendar + Drive (optional, one-time)

1. In [console.cloud.google.com](https://console.cloud.google.com) (same project as your
   Gemini key): **APIs & Services → Library** → enable **Gmail API**, **Google Calendar API**,
   and **Google Drive API**
2. **Credentials → Create OAuth client ID → Desktop app** → download the JSON
3. Save it as `credentials.json` in this folder

The first Google request opens your browser to authorize; the token is cached in
`token.json` afterwards. Both files are gitignored. (If you authorized before Drive
was added, delete `token.json` once to pick up the new scope.)

### Discord (optional, 1 minute)

Channel settings → **Integrations → Webhooks → New Webhook** → copy the URL into
`DISCORD_WEBHOOK_URL` in `.env`. Enables the `discord_send` tool, and `daemon.py`
mirrors due-task reminders to that channel (reaches your phone via the Discord app).

### Background daemon (optional)

```bash
python daemon.py        # in a second terminal
```

Polls `osiris.db` every 60 s and fires a Windows toast when a scheduled task
comes due. Zero API usage.

## Try it

```
you › remember that I'm targeting IBM Quantum roles after graduating in 2027
you › I want to get into Stanford AI in 5 years          ← it will offer to plan it
you › remind me tomorrow 9am to review my PhantomGrid threat model
you › is PromptStrike in sync with GitHub? if not, commit and push it
you › draft replies to my 3 unread emails — I'll review them in Gmail
you › list what's in my StylesOS folder
you › any unread emails about the hackathon?
you › what's on my calendar this week?
you › what do you know about me?
you › /goals
```

Memory, goals, and tasks persist in `osiris.db` across sessions. Due tasks are
announced at startup.

## Tests

```bash
pytest tests/ -q
```

Fully offline — the LLM is stubbed, git tests run against a throwaway temp repo,
and integration tools are verified to fail gracefully without credentials.

## Architecture notes

- **Model-agnostic by design.** The agent speaks a neutral message/tool format
  ([osiris/providers/base.py](osiris/providers/base.py)). Groq, Ollama, and OpenAI
  share one OpenAI-compatible adapter; adding any new provider = one small class.
- **Failover, not fragility.** Gemini rate-limited? The router silently retries the
  same request on Groq, then Ollama. Set priority with `OSIRIS_PROVIDERS` in `.env`.
- **Guarded local access.** `fs_*` (read/list/find) and `git_*` tools are confined to
  allow-listed roots (your Desktop by default; `OSIRIS_ALLOWED_DIRS` to change) — system
  paths are rejected. Writes are further sandboxed to `./workspace/`. The agent is
  instructed to show status before committing and get explicit confirmation before pushing.

## Roadmap

- [x] Gmail + Google Calendar via OAuth (same Google Cloud project as the API key)
- [x] Background daemon: fire due tasks as notifications instead of at-startup
- [x] Model router: Gemini → Groq → Ollama → OpenAI with automatic failover
- [x] Specialized agents with per-agent tool subsets
- [x] Google Drive (search + read) and Discord (send + daemon reminders)
- [x] Web frontend: terminal-style chat with live goals/tasks panel
- [x] Reflection pass: sessions are distilled into long-term memories
      (`/reflect`, on REPL exit, and every 8 web turns in the background —
      deduplicated by embedding similarity)
- [x] CI: pytest on Ubuntu + Windows
- [ ] Knowledge graph over memories (entities + relations, not just embeddings)
- [ ] Notion integration
