"""Benchmark: does schema-constrained-decode repair actually fix broken tool
calls from small local Ollama models, measured against OSIRIS's real tool
registry (not a synthetic toy schema)?

Methodology
-----------
1. BASELINE: send each test prompt through the exact code path OSIRIS uses
   today (OpenAICompatLLM.chat, tools=<real registry specs>) — this already
   includes the existing regex-based rescue hacks in openai_compat.py, so
   "baseline" here means "current OSIRIS behavior," not a stripped-down
   worst case.
2. Classify each result: correct tool selected? required arguments present?
3. REPAIR: for the specific, narrow failure mode this targets — right tool
   picked, but required arguments missing/malformed — re-decode just the
   arguments constrained to that tool's own JSON schema
   (schema_repair.repair_args) and re-check.

This does NOT attempt to fix tool-SELECTION misses (model didn't call any
tool when one was needed) — there's no schema to constrain against without
knowing which tool was intended. That's reported separately, honestly, as
a limitation, not folded into the "fixed" number.

Run:  python -m benchmarks.tool_calling_reliability
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from osiris import config
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.providers import schema_repair
from osiris.providers.base import BaseLLM, LLMResponse
from osiris.providers.openai_compat import OpenAICompatLLM
from osiris.tools import build_registry

OLLAMA_BASE = f"{config.OLLAMA_HOST}/v1"
MODELS = ["llama3.2", "gemma3", "qwen3"]

# 10 target tools (one per test case below) + 5 distractors, so tool
# SELECTION is a real decision, not "the only tool that could possibly fit."
TOOL_SUBSET = {
    "remember", "recall", "schedule_task", "git_status", "git_log",
    "fs_list", "fs_read", "git_commit", "current_datetime", "note",
    "fs_write", "list_tasks", "list_goals", "forget", "web_search",
}

TEST_CASES = [
    {"prompt": "Remember that I prefer dark mode in all my editors.",
     "expected_tool": "remember"},
    {"prompt": "What do you know about my coding preferences?",
     "expected_tool": "recall"},
    {"prompt": "Schedule a reminder to call mom tomorrow at 5pm.",
     "expected_tool": "schedule_task"},
    {"prompt": "Show me the git status of my PromptStrike repo.",
     "expected_tool": "git_status"},
    {"prompt": "What are the last 5 commits in the osiris repo?",
     "expected_tool": "git_log"},
    {"prompt": "List the files in my StylesOS folder.",
     "expected_tool": "fs_list"},
    {"prompt": "Read the README.md file in my Bolt project.",
     "expected_tool": "fs_read"},
    {"prompt": "Commit my changes in the darkdecoder repo with the message 'fix login bug'.",
     "expected_tool": "git_commit"},
    {"prompt": "What's today's date and time?",
     "expected_tool": "current_datetime"},
    {"prompt": "Note that I finished the OSIRIS README rewrite.",
     "expected_tool": "note"},
]

_SYSTEM = ("You are OSIRIS, a personal AI assistant. Use tools proactively "
          "to fulfil the user's request. Be concise.")


class _DummyLLM(BaseLLM):
    """Memory/Planner need an embed-capable LLM to construct; the benchmark
    never calls remember()/recall() for real, so this just needs to exist."""
    name = "dummy"
    def chat(self, messages, system="", tools=None):
        return LLMResponse(text="")
    def embed(self, text):
        return [0.0] * 8


def _build_specs():
    memory = Memory(_DummyLLM(), db_path=":memory:")
    registry = build_registry(memory, Planner(_DummyLLM(), memory))
    return registry, registry.specs(only=TOOL_SUBSET)


def _grade(resp: LLMResponse, expected_tool: str, schema_by_name: dict) -> dict:
    if not resp.tool_calls:
        return {"outcome": "no_tool_call", "tool": None, "args_ok": False}
    tc = resp.tool_calls[0]
    schema = schema_by_name.get(tc.name, {}).get("parameters", {})
    args_ok = not schema_repair.needs_repair(tc.args, schema)
    if tc.name != expected_tool:
        return {"outcome": "wrong_tool", "tool": tc.name, "args_ok": args_ok}
    return {"outcome": "correct" if args_ok else "right_tool_bad_args",
           "tool": tc.name, "args_ok": args_ok}


def run_model(model: str, specs: list, schema_by_name: dict) -> dict:
    llm = OpenAICompatLLM(f"ollama-{model}", OLLAMA_BASE, "ollama", model)
    results = []
    for case in TEST_CASES:
        t0 = time.time()
        try:
            resp = llm.chat([{"role": "user", "content": case["prompt"]}],
                            system=_SYSTEM, tools=specs)
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            results.append({"prompt": case["prompt"], "expected": case["expected_tool"],
                            "outcome": f"error: {type(e).__name__}: {e}", "repaired": None,
                            "seconds": elapsed})
            print(f"  [{model}] {case['prompt'][:50]:50} -> error: {type(e).__name__}: "
                 f"{str(e)[:80]} ({elapsed}s)")
            continue
        baseline = _grade(resp, case["expected_tool"], schema_by_name)
        repaired = None
        if baseline["outcome"] == "right_tool_bad_args":
            tool_schema = schema_by_name[baseline["tool"]]["parameters"]
            fixed_args = schema_repair.repair_args(
                OLLAMA_BASE, "ollama", model, baseline["tool"], tool_schema, case["prompt"])
            repaired = "fixed" if fixed_args and not schema_repair.needs_repair(fixed_args, tool_schema) \
                else "still_broken"
        results.append({"prompt": case["prompt"], "expected": case["expected_tool"],
                        **baseline, "repaired": repaired,
                        "seconds": round(time.time() - t0, 1)})
        print(f"  [{model}] {case['prompt'][:50]:50} -> {baseline['outcome']:20} "
             f"repair={repaired} ({results[-1]['seconds']}s)")
    return {"model": model, "results": results}


def summarize(all_results: list) -> str:
    lines = ["", "=" * 70, "SUMMARY", "=" * 70]
    for run in all_results:
        model, results = run["model"], run["results"]
        n = len(results)
        correct = sum(1 for r in results if r["outcome"] == "correct")
        bad_args = sum(1 for r in results if r["outcome"] == "right_tool_bad_args")
        fixed = sum(1 for r in results if r["repaired"] == "fixed")
        no_call = sum(1 for r in results if r["outcome"] == "no_tool_call")
        wrong = sum(1 for r in results if r["outcome"] == "wrong_tool")
        errors = sum(1 for r in results if r["outcome"].startswith("error:"))
        before_valid_pct = round(100 * correct / n, 0) if n else 0
        after_valid_pct = round(100 * (correct + fixed) / n, 0) if n else 0
        if errors:
            lines.append(f"{model:12} {errors}/{n} calls errored outright (see results.json "
                        f"for messages) — this model likely can't do tool-calling at all here")
        lines.append(
            f"{model:12} baseline valid: {correct}/{n} ({before_valid_pct:.0f}%)  "
            f"| right-tool-bad-args: {bad_args} (repaired {fixed}/{bad_args or 1})  "
            f"| no_tool_call: {no_call}  wrong_tool: {wrong}  "
            f"| after repair: {correct + fixed}/{n} ({after_valid_pct:.0f}%)")
    return "\n".join(lines)


def main():
    registry, specs = _build_specs()
    schema_by_name = {s["name"]: s for s in specs}
    print(f"Testing {len(TEST_CASES)} cases x {len(MODELS)} models "
         f"against {len(specs)} tools ({', '.join(sorted(TOOL_SUBSET))})\n")

    all_results = []
    for model in MODELS:
        print(f"--- {model} ---")
        all_results.append(run_model(model, specs, schema_by_name))

    report = summarize(all_results)
    print(report)

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
