# OSIRIS Handbook

Personal AI operating system. Model-agnostic brain, 6 specialized agents, real access to your
filesystem/git/GitHub/Gmail/Calendar/Drive/Discord, persistent memory, proactive daemon.

---

## 1. Running it

```
osiris              REPL — talk to it in any terminal
osiris serve        web UI at http://localhost:8765
osiris daemon       background: reminders + daily routines
```

All three share the same `osiris.db` — anything one learns, the others know. Run all three if
you want the full system (REPL/web for talking to it, daemon for it to work while you don't).

`osiris` is installed globally via `osiris.cmd` copied to
`%LOCALAPPDATA%\Microsoft\WindowsApps\`. If you ever move the OSIRIS folder, update
`OSIRIS_HOME` inside that copied `.cmd` file.

---

## 2. REPL commands

```
/goals      active goals + milestone progress
/tasks      pending scheduled reminders
/status     health check — providers, integrations configured, memory/goal/task counts
/model      which provider answered last + who's cooling down right now
/loop <seconds> <prompt>   repeat a prompt on an interval until Ctrl+C
/reflect    distill this session into long-term memory right now
/clear      wipe this session's conversation (long-term memories are kept)
/help       this list
/quit       exit (auto-reflects if the session had ≥3 turns)
```
`/loop` is a foreground, immediate repeat — distinct from the daemon's scheduled routines,
which run in the background once a day regardless of whether the REPL is open. Use `/loop` for
"watch this while I work" (e.g. `/loop 300 check unread mail`); use routines for "every
morning, unattended."
Anything else is sent straight to the agent. Web UI has the same reset as a **CLEAR** button
in the header (`POST /api/clear`).

**Readability**: replies wrap to your actual terminal width with a hanging indent under "- "
bullets, so a wrapped line's continuation aligns under the text instead of restarting at
column 0 and looking like a new item. The "you ›" prompt, tool-call echoes, agent-routing tag,
and provider tag use real ANSI color (dim gray for metadata, amber for your input) — actual
terminal formatting, not markdown syntax, and automatically off if stdout isn't a real
terminal (piped/redirected output stays clean).

## 2b. If something breaks — the playbook

Realistic expectation: nothing running on free-tier models across a fully autonomous tool
loop will *never* break. What's built is making failures soft and recoverable instead of
silent or catastrophic:

- **A build/task looks stuck or wrong** → say `continue`. Shared conversation history means
  the next turn (even on a different provider, after a failover) picks up with full context
  of what was already tried — this is the exact recovery that worked live when Mistral
  thrashed on a dependency issue and switching models resolved it.
- **A reply looks broken, repetitive, or the conversation feels "off"** → `/clear` (REPL) or
  **CLEAR** button (web). Resets the session only — remembered facts, goals, and tasks are
  untouched, verified live (goal/task/memory counts stayed identical across a clear).
- **It hits the step limit** → the stop message now tells you exactly what to do next
  ("say 'continue' to resume with a fresh step budget") instead of a dead end.
- **A weak model keeps retrying the same failing tool call** → a circuit breaker in
  `agent.py` now injects a corrective nudge after 3 identical repeats, forcing a different
  fix or a clean "I can't resolve this" instead of silently burning the whole step budget —
  built directly from watching Mistral call `run_project_command` far more than the "3
  attempts" persona guidance intended, unchecked, until it hit the hard cap.
- **Nothing responds at all** → check `osiris serve`/`osiris daemon` are actually running in
  their terminals; a closed terminal window kills that process.

None of this requires touching code — it's all built into the running system now.

---

## 3. How a message becomes a reply

```
your message
   → pick_agent(text)          keyword match against 8 profiles, weighted + tie-broken
   → recall relevant memories  top-3 by similarity (or keyword search if embeddings are off)
   → ModelRouter.chat()        tries providers in chain order, first success wins
   → tool-call loop            up to profile.max_steps iterations
   → plain-text reply
```

Everything happens inside one shared `Agent.history` list — switching agents mid-conversation
doesn't lose context.

### 3b. Reasoning before acting

The system prompt now tells every agent to work through what the user actually needs, which
tools in what order, and what could go wrong — *before* acting — but to never show that
reasoning, and never open a reply with "Let me think" or similar.

**Be honest with yourself about what this is**: it's a prompt instruction, not a native hidden
reasoning-token feature (unlike models with a real thinking mode). It relies on the model
choosing to comply — Gemini and Groq's `oss-120b` follow it well; a weaker fallback (Ollama,
in particular) may occasionally leak "First, I will..." text despite the instruction. If you
want genuine built-in chain-of-thought rather than a prompted approximation of it, OpenRouter's
free tier includes reasoning-native models (e.g. `deepseek/deepseek-r1:free`) — swap
`OSIRIS_OPENROUTER_MODEL` in `.env` to try one; you already have `OPENROUTER_API_KEY` set.

### 3c. Output formatting — no markdown, anywhere

Neither the REPL nor the web UI renders markdown (both print raw text — confirmed by reading
`frontend/index.html`, which uses `textContent`, not an HTML renderer). So `**bold**` and
`# headers` were never a style choice showing up nicely — they were literal asterisks and
hashes cluttering every reply. Two layers fix this:

1. **Persona instruction**: the system prompt tells every agent plain text only, no markdown,
   and to prefer a natural conversational answer over a labeled "report" dump (the "Profile
   Summary" with a `**Field:**` per line was Groq doing more than actually asked for a casual
   "tell me about myself").
2. **Deterministic strip** (`osiris/agent.py::_strip_markdown`): because persona instructions
   are prompts, not guarantees — Groq's `oss-120b` kept using `**` even when told not to,
   verified live — every reply is regex-stripped of `**bold**`, `__bold__`, and `# headers`
   before it reaches you, regardless of which provider answered or whether it complied.

### 3d. Persistent identity — always present, not just recalled

`OSIRIS_USER_PROFILE` in `.env` is injected into *every* agent's system prompt on *every* turn
— unlike `remember`ed facts, which only surface if `recall`'s top-3 similarity search happens
to rank them. Edit it directly whenever your situation changes (new project, new target role);
no code touch, no restart-triggering migration, just a line of text read fresh each run.

---

## 4. The 8 agents

Each agent is a **persona + a restricted tool subset + routing keywords**. Smaller tool lists
= cheaper calls + sharper tool selection, especially on weaker fallback models.

| Agent | Triggers on | Tools | Notes |
|---|---|---|---|
| **architect** | "build me", "build a", "scaffold", "architecture", "full project", "high end" | scaffold_project, fs_*, git_init/status/commit/push, github_create_repo/pr, **run_project_command** | Designs + writes an entire project, then **mandatorily verifies it** with a real `run_project_command` run before claiming done — up to 3 fix attempts on failure (fs_edit + re-run), see §5b. Defaults to `OSIRIS FILES/`. Highest priority tiebreak among build-ish phrasing. |
| **coding** | "github", "repo", "commit", "push", "diff", "pr ", "desktop", "sync" | full git/github toolset + fs_* + scaffold_project + **run_project_command** + **analyze_image**, **index_folder**/**search_docs**, **write_pdf** | General repo work: status checks, explaining a codebase, modifying files, triaging issues, actually running a repo's tests/build/lint. |
| **redteam** | "pentest", "exploit", "recon", "ctf", "bug bounty", "sqli/xss", "nmap", "vapt" | fs_*, scaffold_project, git, **run_recon**, **cve_lookup**, **analyze_image**, **write_pdf** | Bug-hunt log tracking, PoC scaffolding, live recon (nmap/whois/dig/nslookup/ping/curl) against **authorized targets only**, live CVE data from NVD. |
| **blueteam** | "blue team", "detection", "incident response", "soc ", "siem", "ioc" | fs_*, scaffold_project, git, **run_recon** (IOC lookups), **cve_lookup**, **analyze_image**, **write_pdf** | Detection-rule drafting, IR playbooks, no live SIEM connection. |
| **aisec** | "prompt injection", "jailbreak", "ai security", "llm security", "guardrail", "gcg", "crescendo" | fs_*, scaffold_project, git | LLM red-teaming methodology, adversarial test design — the PromptStrike/DarkDecoder domain. No live model API calls. |
| **email** | "email", "mail", "inbox", "unread", "draft", "cold email", "outreach", "application" | gmail_search/read/send/create_draft/reply, **web_search**, **track_application/list_applications/update_application_status**, schedule_task | Never sends without your explicit confirmation; always prefers drafts. Cold outreach is always one *personalized* draft per company, never a templated blast — see §14. |
| **learning** | "learn", "study", "goal", "roadmap", "milestone" | create_plan, list_goals, complete_milestone, schedule_task, list_tasks, complete_task | Long-horizon goal planning with due-dated milestones. |
| **research** | "research", "investigate", "summarize", "drive", "docs", "search the web", "look up" | write_file/read_file/list_files, drive_search/read, fs_*, **web_search**, **read_pdf**, **analyze_image**, **index_folder**/**search_docs**, **write_pdf** | Synthesizes findings from the live web + Drive + local indexed docs + memory into workspace notes, or a generated PDF. |
| **personal** | fallback (no keyword match) | **all 57 tools** | The expensive path — use it for genuinely mixed-domain asks, not routine ones. |

Routing is pure keyword weighting in [agents.py](osiris/agents.py) — `pick_agent()` sums
`hits × keyword_weight` per profile, ties broken by `priority`. No ML classifier, so it costs
nothing and never fails.

---

## 5. Tool reference (57 tools)

**Memory** — `remember`, `recall`, `list_memories`, `forget`, `current_datetime`
**Journal** — `note` (quick-capture a timestamped line into workspace/journal/YYYY-MM-DD.md), `read_journal`
**Planning** — `create_plan`, `list_goals`, `complete_milestone`
**Scheduler** — `schedule_task`, `list_tasks`, `complete_task`
**Workspace (sandboxed to `./workspace`)** — `write_file`, `read_file`, `list_files`
**Desktop filesystem (sandboxed to `ALLOWED_DIRS`)** — `fs_list`, `fs_read`, `fs_find`, `fs_write`, `fs_edit`, `read_pdf` (extract text from a PDF — resumes, JDs, papers; needs `pdfplumber`, scanned/image PDFs won't extract)
**Project scaffolding** — `scaffold_project` (writes an entire project tree in one call)
**Local git** — `git_list_repos`, `git_init`, `git_status`, `git_diff`, `git_log`, `git_commit`, `git_push`, `git_sync_check`
**GitHub** — `github_create_repo`, `github_create_pr`, `github_list_repos`, `github_list_issues`, `github_create_issue`
**Recon (authorized targets only)** — `run_recon` (nmap/whois/dig/nslookup/ping/curl)
**Project ops** — `run_project_command` (pytest/npm-test/npm-build/npm-lint/ruff/mypy, one repo at a time), `cve_lookup` (live NVD database, no key needed)
**Web** — `web_search` (Tavily, live results — needs `TAVILY_API_KEY`)
**Job tracking** — `track_application`, `list_applications`, `update_application_status`
**Gmail** — `gmail_search`, `gmail_read`, `gmail_send`, `gmail_create_draft`, `gmail_reply`
**Calendar** — `calendar_list_events`, `calendar_create_event`
**Drive** — `drive_search`, `drive_read`
**Discord** — `discord_send`
**Vision** — `analyze_image` (describe/answer questions about a screenshot or photo — needs `GEMINI_API_KEY`, the only configured provider here with vision)
**Local document index** — `index_folder` (embeds a folder's text/code files, skips unchanged files on re-run), `search_docs` (semantic search over what's been indexed)
**Documents** — `write_pdf` (generate a PDF from scratch in the workspace — heading/bullet/paragraph plain text, no markdown needed)

### 5b. The generate → test → fix loop (architect)

The highest-leverage quality fix available on free-tier models: never trust one-shot code
generation, verify against real test output instead. Architect's workflow is now:

```
scaffold_project → run_project_command (pytest, usually)
   ↓ fails                              ↓ passes
   read the actual error                git_init + git_commit
   fs_edit the specific file
   run_project_command again  ─────────→ (up to 3 attempts)
```

If it can't get green after 3 attempts, it's instructed to say so plainly and show the last
failure — not silently claim success. This is a persona-level instruction, so a weak model can
still ignore it — observed live: Mistral called `run_project_command` far more than 3 times on
a stubborn dependency issue (bcrypt/sqlmodel install failures) without stopping, eventually
hitting the hard step cap. Two things now catch that:

1. **Circuit breaker** (`agent.py`, code-level, not persona-level): after 3 consecutive calls
   to the *same* tool, a corrective message is force-injected into history telling the model
   to stop retrying the same way and either make one genuinely different fix or give up
   cleanly. Doesn't guarantee success on a weak model, but reliably shortens the thrash.
2. **Architect's step budget raised 20 → 30** — a real multi-file build with a dependency
   snag genuinely needs the room; the circuit breaker keeps the extra budget from just being
   more room to thrash in.

If it still hits the cap, `continue` in the same conversation resumes with a fresh step
budget and full context of what was already tried — this is what actually recovered the live
FastAPI/JWT build (Mistral got stuck, "continue... different model" picked it up on Groq and
finished it). See §2b.

### Confirm-before-acting tools (persona-enforced, not hard-gated)
`fs_write`, `fs_edit`, `git_commit`, `git_push`, `github_create_repo`, `github_create_pr`,
`gmail_send`, `run_recon`, `run_project_command` — the agent is instructed to show you the
exact action and get an explicit go-ahead in chat before calling these. This is a
prompt-level convention, not a code gate — if you ever see one of these fire without you
having agreed, that's a persona regression worth flagging.

`gmail_create_draft`/`gmail_reply` are NOT in this list — they only ever write to Drafts,
never send, so there's nothing irreversible to confirm.

### Hard-gated (code-enforced, cannot be argued around)
- `fs_*` / `git_*` / `run_recon` / `run_project_command` targets outside `ALLOWED_DIRS` →
  `PermissionError`, always
- `write_file`/`read_file` outside `./workspace` → `PermissionError`, always
- `run_recon` on a tool not in the allowlist, a flag not in the per-tool safe set, or a target
  that isn't localhost/private/explicitly authorized → hard error, always
- `run_project_command` on a command not in its allowlist (pytest/npm-test/npm-build/
  npm-lint/ruff/mypy — no arbitrary scripts), or `extra_args` containing shell metacharacters
  → hard error, always
- `scaffold_project` writing outside its own project folder (`../` escapes) → hard error

---

## 6. Provider chain (model-agnostic failover)

```
gemini → groq → cerebras → mistral → openrouter → github → ollama → openai
```

Configured via `OSIRIS_PROVIDERS` in `.env`; unconfigured providers are silently skipped, so
listing all 8 costs nothing. On any error (rate limit, timeout, malformed response) the
router cools that provider down (90s for rate limits, 30s otherwise) and retries the next one
in the chain — your conversation never just stops. `[provider · model]` under every reply
tells you who actually answered.

- **Gemini 2.5 Flash** — primary, best tool-use quality, 1500 req/day free (resets midnight
  Pacific ≈ 12:30pm IST)
- **Groq `gpt-oss-120b`** — first fallback, good tool-use
- **Cerebras / Mistral / OpenRouter / GitHub Models** — additional free-tier fallbacks (only
  active once you add their keys to `.env`)
- **Ollama** — local, zero cost, auto-picks your best pulled model (`qwen3` > `qwen2.5` >
  `llama3.3` > `llama3.1` > `phi4` > `mistral` > `llama3.2`; `gemma` is skipped — no tool
  support)
- **OpenAI** — last resort if you ever add a paid key

Debug mode: `OSIRIS_DEBUG=on` in `.env` shows the raw failover chatter (which provider
tried, what error). Off by default for clean output.

---

## 7. Memory system

SQLite at `osiris.db`, six tables: `memories`, `goals`/`milestones`, `tasks`, `routine_runs`,
`applications`.

- **Auto-store**: the agent calls `remember` on its own for durable facts, unprompted.
- **Auto-recall**: every turn searches the top 3 relevant memories before replying — this is
  how it stays personalized without you repeating context.
- **Vector search** when `OSIRIS_EMBEDDINGS=on` (costs a few Gemini embedding calls — trivial
  at personal memory-count scale); falls back to keyword search when off. Now **on** — keyword
  search was demonstrably missing obvious facts (a query like "tell me about myself" would
  return only a memory sharing a literal word with the query, nothing else).
- **Reflection**: at REPL `/quit` or every 8 web-UI turns, a pass distills the session into
  0-5 durable facts, deduplicated against existing memories at ≥0.90 cosine similarity.
- **`forget` exists now** — `list_memories` shows every stored fact with its id, `forget(id)`
  permanently deletes one. Confirm-first, same convention as the other destructive tools; not
  hard-gated, so a misbehaving model could in principle skip confirming — same caveat as
  everything else in this list.

---

## 8. Proactive daemon

If you're running `osiris bot` (§9b), you don't need this running separately — the bot polls
the same cycle internally (`daemon.poll_once`, shared code, not a reimplementation) and DMs you
directly. Run `osiris daemon` standalone only if you want toast/webhook notifications without
the Discord bot's chat surface.

`osiris daemon` polls every 60s for two things:

1. **Scheduled task reminders** — Windows toast (`plyer`) + Discord webhook mirror if configured.
2. **Routines** (`routines.json`) — runs the full agent at scheduled times, saves the reply to
   `workspace/briefings/YYYY-MM-DD-{name}.md`, fires a toast + Discord notification. Each
   routine runs with cleared agent history — independent context, no bleed between routines.

Each routine now declares its `agent` explicitly (`"agent": "coding"`, etc.) rather than
relying on keyword routing. This matters: keyword-routing a pre-written prompt is fragile — the
original `morning-brief` prompt contained the substring "mail" (inside `gmail_search`) and a
literal `MAIL` header, both of which are `email`-agent keywords, so it was silently landing on
the `email` profile — which has no `git_list_repos` or `calendar_list_events`. The REPOS and
CALENDAR sections of your daily brief were likely empty this whole time. Fixed by making the
agent explicit; omit `"agent"` and it defaults to `personal` (full toolset, safest default).

Optional `"days": ["Sun"]` restricts a routine to specific weekdays (3-letter, `%a` format) —
omit for the default daily cadence.

Default routines:
- **08:30 morning-brief** (`personal`) — dirty repos, unread mail (internships/hackathons/
  deadlines only), next 2 days of calendar, pending tasks
- **21:30 evening-repo-check** (`coding`) — flags any repo with uncommitted changes or
  unpushed commits
- **09:00 Sunday security-digest** (`redteam`) — 2-3 CVEs via live `cve_lookup` rotating
  through prompt injection / web vuln classes / vendor CVEs / LLM framework vulns, plus 2-3
  notable items via `web_search` scoped to red-team/blue-team/AI-security news

Add your own by appending to `routines.json`:
```json
{"name": "pre-interview", "time": "09:00", "agent": "learning", "prompt": "Give me a 3-question mock interview on quantum computing and grade my answers"}
```

The daemon builds the LLM/agent lazily — if no routine has fired and no task is due, it costs
**zero API quota** while idle.

---

## 9. Web UI (`osiris serve`)

FastAPI app on `http://localhost:8765`. Same agent/memory/tools as the REPL.

| Endpoint | What |
|---|---|
| `POST /api/chat` | send a message, get `{reply, agent, tools, provider}` |
| `POST /api/reflect` | manually trigger reflection |
| `POST /api/clear` | reset the conversation (memories/goals/tasks untouched) |
| `GET /api/state` | providers, goals, tasks, memory count |
| `GET /` | serves `frontend/index.html` |

Background reflection runs automatically every 8 chat turns, non-blocking (separate thread).

**Look**: JetBrains Mono (Google Fonts, falls back to Cascadia/Consolas/monospace if offline),
CRT-green terminal aesthetic — your messages render amber (`--amber`), OSIRIS's render green
with a faint glow (`text-shadow`), so a conversation is scannable at a glance instead of one
undifferentiated color block. A **CLEAR** button sits in the header next to the provider chain.

---

## 9b. Mobile dispatcher (`osiris bot`)

A real Discord bot (`discord_bot.py`) — not the send-only webhook from §8, a genuine bidirectional
chat. Message OSIRIS from the Discord app on your phone, from anywhere with internet (mobile
data works, no VPN/tunnel to your home network needed).

**Security model**: only DMs from `OSIRIS_DISCORD_ALLOWED_USERS` (your Discord user id) get a
response. Server messages are ignored outright — DM-only surface, smallest exposure. Anyone
else who DMs the bot gets silence: no reply, no hint it exists. This is a new *front door*, not
new *permissions* — every tool's own guardrails (`run_recon`'s authorization gate,
`run_project_command`'s allowlist, confirm-before-act persona rules) apply identically here.

**Setup**:
1. discord.com/developers/applications → New Application → Bot tab → enable **Message Content
   Intent** under Privileged Gateway Intents → Reset Token, copy it → `.env`: `DISCORD_BOT_TOKEN=`
2. Discord app → User Settings → Advanced → enable Developer Mode → right-click your own
   username → Copy User ID → `.env`: `OSIRIS_DISCORD_ALLOWED_USERS=<your id>` (comma-separated
   for more than one)
3. OAuth2 → URL Generator → scope `bot`, permissions Send Messages + Read Message History (no
   admin) → invite it to a small private server with just you, then DM it directly
4. `osiris bot` (needs `discord.py` — already in `requirements.txt`)

**One process, not two**: `osiris bot` also runs the daemon's reminder/routine poll cycle
internally (`daemon.poll_once` — shared code, not duplicated) and DMs you directly when
something fires, on top of whatever toast/webhook `notify()` already does. You no longer need
`osiris daemon` running separately just to get proactive behavior — verified live: a due
reminder correctly fires exactly once and doesn't double-notify on the next poll.

**The honest limit, unavoidable**: your PC has to actually be on and this process running.
OSIRIS is a program on your machine, not a hosted cloud service — if the PC's off or asleep,
the bot goes offline and messages just wait until it's back.

---

## 10. Folder map

```
OSIRIS/                    the app itself
  osiris/                  package: agent.py, agents.py, router.py, tools.py,
                            memory.py, planner.py, reflection.py, config.py,
                            google_services.py, providers/
  workspace/                write_file/read_file sandbox + briefings/
  tests/                    67 offline pytest tests
  osiris.db                 SQLite: memories, goals, tasks, routines
  .env                      your keys — OSIRIS never writes to this file
  credentials.json /
  token.json                 Google OAuth (Gmail/Calendar/Drive)
  routines.json              daemon's daily-prompt schedule

OSIRIS FILES/               [Desktop, separate from OSIRIS/] — default
                            destination for scaffold_project. Everything the
                            architect/coding/redteam/etc. agents BUILD lands
                            here unless you name another folder. Nothing else
                            writes here automatically — it's purely an output
                            drop zone, not a working directory OSIRIS reads
                            from on its own.
```

`ALLOWED_DIRS` (fs_*/git_*/run_recon reach) defaults to your whole `Desktop` — including
`OSIRIS FILES/` and every other project folder (DarkDecoder, PromptStrike, SOC-PARALLAX,
PhantomGrid, StylesOS, etc.). Override with `OSIRIS_ALLOWED_DIRS=path1;path2` in `.env`.

---

## 11. `.env` reference

```env
# providers (add any subset — unconfigured ones are skipped)
GEMINI_API_KEY=
GROQ_API_KEY=
CEREBRAS_API_KEY=
MISTRAL_API_KEY=
OPENROUTER_API_KEY=
GITHUB_MODELS_TOKEN=
OPENAI_API_KEY=

# integrations
GITHUB_TOKEN=                    # repo scope — git push, github_* tools
DISCORD_WEBHOOK_URL=             # daemon notifications mirror here
DISCORD_BOT_TOKEN=               # osiris bot — mobile dispatcher, see §9b
OSIRIS_DISCORD_ALLOWED_USERS=    # your Discord user id(s), comma-separated

# git identity for git_commit
OSIRIS_GIT_NAME=Pyhroff
OSIRIS_GIT_EMAIL=215876416+Pyhroff@users.noreply.github.com

# identity — injected into every agent's system prompt, always present (see §3d)
OSIRIS_USER_PROFILE=3rd-year BTech Cybersecurity student at IIIT Kottayam...

# security
OSIRIS_AUTHORIZED_TARGETS=       # comma-separated hosts run_recon may hit beyond localhost/private

# web search
TAVILY_API_KEY=                  # free, no card: tavily.com — powers the web_search tool

# tuning
OSIRIS_EMBEDDINGS=on             # on = vector recall (costs a few Gemini calls), off = keyword search
OSIRIS_DEBUG=off                 # on = show provider failover chatter
OSIRIS_PROVIDERS=gemini,groq,cerebras,mistral,openrouter,github,ollama,openai
OSIRIS_ALLOWED_DIRS=             # override fs_*/git_*/run_recon reach; default = Desktop
```

Never write secrets via OSIRIS's tools — it will always tell you to edit `.env` by hand and
restart. Changes to `.env` require a restart to take effect (no hot reload).

---

## 12. Security model, honestly

- **Filesystem**: hard-sandboxed to `ALLOWED_DIRS` (Desktop by default) for read/write/git/recon.
  Cannot touch anything outside it, `~` expansion included. The Desktop root is resolved via
  the actual Windows registry Known Folder value (same thing Explorer reads), not a hardcoded
  guess — matters if OneDrive redirects your Desktop, since `~/Desktop` and `~/OneDrive/Desktop`
  can both physically exist and a model that ignores "never guess this path" can silently land
  a file in the wrong (stale) one. Observed live, now fixed at the config level, not just prompted.
- **Google API errors**: Gmail/Calendar/Drive failures (expired token, quota, bad permission)
  surface as one clean sentence, not a raw JSON body dump — including an actionable message
  when a refresh token is revoked/expired ("delete token.json and run `osiris` once to
  re-authorize"), which matters on a long-running bot/daemon process more than a REPL you
  restart daily.
- **Empty-response recovery**: if a provider returns neither text nor a tool call (observed
  live with Gemini on casual/typo'd requests — "chec my mails" silently returned nothing twice
  in a row before a clearer phrasing worked), the agent now nudges and retries within the same
  turn instead of surrendering with a dead-end "(no response)".
- **Confirm-before-act**: enforced by persona instructions the LLM is told to follow, not by
  a hard code gate (except `run_recon`'s target/flag checks, which are hard-gated). A
  misbehaving model *could* skip the confirmation step — watch for that, especially on weaker
  fallback providers.
- **`run_recon`**: hard-gated allowlist (6 tools, no exploitation frameworks), hard-gated
  per-tool flags (no file-write or script-exec flags), hard-gated targets (localhost/private
  or your explicit `OSIRIS_AUTHORIZED_TARGETS` list).
- **Web search exists now** (`web_search`, Tavily) — but it needs `TAVILY_API_KEY` in `.env`
  (free, no card, at tavily.com) or every call returns a clear "not configured" error instead
  of silently failing.
- **Cold outreach**: `email` agent will only draft — never send — and writes one personalized
  email per company, never a templated blast. It can now use `web_search` to find a company's
  already-*public* careers/HR contact if you don't already have one (and tells you what it
  found before drafting to it, since search can be wrong) — this is search, not scraping, and
  it's still one company/one draft/one confirm at a time. It will never auto-draft to a batch
  of companies without you naming each one. See §14.
- **Free-tier code quality**: expect solid architecture/structure/git hygiene from
  `scaffold_project`, but treat the actual logic as a first draft, not ship-ready — these are
  free models, not Opus-tier reasoning.

---

## 13. Quick reference — day-to-day phrasing

```
"check all my repos"                          → coding: git_list_repos, one call
"commit PromptStrike with message X"           → coding: shows status, confirms, commits
"push it"                                      → coding: confirmed, inline-token push
"build me a [X]"                               → architect: scaffold_project → git_init → commit
"put it on github as private"                  → architect/coding: github_create_repo
"check unread emails"                          → email: gmail_search 'is:unread'
"draft a reply saying X"                       → email: gmail_reply, saved to Drafts, never sent
"what's on my calendar this week?"             → personal (calendar tools live there too)
"I want to get X by [date], make a plan"       → learning: create_plan with milestones
"help me plan recon for this bug bounty"       → redteam: log tracking + run_recon (authorized)
"draft an incident response playbook"          → blueteam: IR notes
"design an adversarial jailbreak test suite"   → aisec: eval harness scaffolding
"what's the latest on [X]"                     → research/personal: web_search
"draft outreach to Cyberops, hr@cyberops.com"  → email: personalized draft + track_application
"what's my application pipeline look like?"    → email: list_applications
"what does this screenshot say?"               → analyze_image (needs GEMINI_API_KEY)
"index my StylesOS folder"                     → index_folder, then search_docs works on it
"find the file that handles auth"              → search_docs (after index_folder)
"turn this into a PDF"                         → write_pdf, saved to workspace/
```

---

## 14. Cold outreach & application tracking

This is a **draft-and-track assistant, not a scraper or a mass-mailer** — by design, for your
own benefit as much as anything else: identical templated blasts get Gmail-flagged as spam
(hurting deliverability for the applications you actually care about) and hiring teams can
spot a mail-merge from a mile away.

**What it does:**
1. You name the company. If you already have a contact, give it; if not, it `web_search`es for
   the company's public careers/HR contact and shows you what it found before using it —
   search-based discovery of already-public info, not scraping, and it always confirms before
   drafting to an address it found itself.
2. It also searches the company generally, so the draft references something real and
   specific, not a name-swapped template.
3. `gmail_create_draft`/`gmail_reply` — always a draft, never sent without you saying so.
4. `track_application` logs it (company, role, contact, status, date).
5. Offers a `schedule_task` follow-up reminder ~2 weeks out.

This is strictly one-company-at-a-time. Asking for outreach to "these 10 companies" gets you
10 individual searches/drafts/confirms in sequence, never a silent batch blast.

**Managing your pipeline:**
```
"what's my application pipeline look like?"        → list_applications
"mark the Cyberops application as interview stage"  → update_application_status
"any applications I haven't heard back on?"         → list_applications (filter by status)
```
