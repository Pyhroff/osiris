# Tool-calling reliability: local Ollama models, measured

Run: `python -m benchmarks.tool_calling_reliability` — 10 realistic prompts
against OSIRIS's real tool registry (15 tools: 10 targets + 5 distractors),
through the exact code path OSIRIS uses in production
(`OpenAICompatLLM.chat`, not a mock). Raw results: `results.json`.

## Headline numbers

| Model | Tool+args correct | Failure mode | Avg latency/call |
|---|---|---|---|
| **qwen3** | 10/10 (100%) | — | ~69s (37–155s, CPU-only) |
| **llama3.2** | 7/10 (70%) | 3× picked a plausible wrong tool | ~4.6s |
| **gemma3** | 0/10 | Ollama rejects it outright — no tool support | ~2s (instant reject) |

## What each finding actually means

**gemma3 can't do tool-calling on Ollama, period.** Every call failed in ~2s
— too fast to be a timeout. Confirmed directly: `curl` against
`/v1/chat/completions` with a `tools` payload returns
`"registry.ollama.ai/library/gemma3:latest does not support tools"`. This
matches an existing code comment in `osiris/providers/__init__.py` ("gemma is
absent [from `_OLLAMA_RANK`] on purpose") — this benchmark independently
verified that assumption live rather than just trusting the comment.

**qwen3 is the most accurate model tested, but impractically slow on CPU-only
hardware for an interactive assistant.** It's a "thinking" model — one spot
check showed 568 tokens of chain-of-thought before a 32-token answer. Ollama's
*native* API can suppress that (`think: false`): verified 100s → 10s on a
direct comparison, same correct answer. But OSIRIS calls Ollama through the
*OpenAI-compatible* endpoint (one shared adapter class for 6 providers,
by design — see `openai_compat.py`), and that endpoint silently ignores
`think: false` (verified: still ~78s with it set). Fixing the latency for
real would mean giving Ollama its own native-API code path, which is a
real architecture decision, not folded into this pass — flagged, not done.

**llama3.2's 3 failures are a real, partially-fixable tool-description
ambiguity, not noise.** It confused `remember`→`fs_write`, `recall`→`web_search`,
and `note`→`fs_write` — consistently picking the generic "write/save" tool
over the more specific one. Ruled out the test harness's system prompt as the
cause first (re-ran the 3 failing cases through OSIRIS's actual, full system
prompt, including its explicit "use `remember` for facts" instruction —
still failed the same way). Tried tightening descriptions:

- `recall`'s description → explicitly "internal memory, NOT files or the
  web": **fixed**, verified by re-test.
- `remember`/`note` vs `fs_write`, including an explicit negative constraint
  added to `fs_write`'s own description ("do NOT use this for facts about
  the user or notes"): **did not fix it**, even after two rounds. This
  looks like a genuine llama3.2 reasoning limitation around this specific
  ambiguity, not a documentation gap — reported honestly rather than
  forcing more prompt-tuning to manufacture a clean result.

Shipped: `recall`'s tightened description (`osiris/tools.py`) — the one
change with a proven before/after. Did not touch `remember`/`note`/`fs_write`
descriptions further, since the attempted fix didn't demonstrably help and
I'd rather leave working code alone than ship an unverified guess.

## `schema_repair.py`: built, tested, not yet proven

`osiris/providers/schema_repair.py` re-decodes a tool call's arguments
through Ollama's schema-constrained JSON mode when the model picked the
right tool but produced invalid/incomplete arguments — the exact failure
mode `openai_compat.py`'s existing regex-based rescue hacks
(`_rescue_pseudo_calls`, `_rescue_json_call`) were written for. It's unit
tested (6 tests, offline/mocked) and the mechanism itself is verified to
work in isolation (`response_format: json_schema` confirmed live against
all 3 models on the exact endpoint OSIRIS uses).

What it hasn't done yet: fire for real. Across all 30 calls in this
benchmark, zero landed in "right tool, bad args" — models either got it
right, picked an entirely different tool, or (gemma3) couldn't attempt a
call at all. The mechanism is real and tested; its practical value against
this specific failure mode is untested, not proven. A harder, more
adversarial test set (multi-field schemas, more ambiguous phrasing) would
be needed to actually validate it before claiming it fixes anything.

## Changes shipped from this pass
- `osiris/tools.py`: `recall`'s description tightened (verified fix)
- `osiris/providers/__init__.py`: `_OLLAMA_RANK` comment documents the
  measured qwen3 speed/accuracy tradeoff and why the default wasn't
  reordered (measured on one CPU-only machine; reordering the default for
  everyone on n=1 hardware data would trade a real accuracy edge for a
  speed guess that might not generalize to GPU setups)
- `.env.example`: documents `OSIRIS_OLLAMA_MODEL` as the fix if this exact
  slow-qwen3 situation is hit — the override already existed, it just
  wasn't discoverable for this specific problem
- `osiris/providers/schema_repair.py` + `benchmarks/tool_calling_reliability.py`:
  new, both tested, neither is a "fix" claim on its own yet
