"""OSIRIS test suite — fully offline (LLM stubbed), machine-independent.

Run:  pytest tests/ -q
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from osiris import config
from osiris.agent import Agent
from osiris.agents import PROFILES, pick_agent
from osiris.memory import Memory
from osiris.planner import Planner
from osiris.providers import BaseLLM, LLMResponse, ToolCall
from osiris.providers.gemini import _to_gemini_contents
from osiris.providers.openai_compat import _to_openai_messages
from osiris.router import ModelRouter
from osiris.tools import build_registry


class StubLLM(BaseLLM):
    name = "stub"

    def chat(self, messages, system="", tools=None):
        return LLMResponse(text=(
            '[{"title": "Learn linear algebra", "detail": "MIT 18.06", "due": "2026-09-01"},'
            ' {"title": "Publish first paper", "due": "2027-06-01"}]'))

    def embed(self, text):
        v = [0.0] * 64
        for i, c in enumerate(text.lower()):
            v[(ord(c) * 31 + i) % 64] += 1.0
        return v


@pytest.fixture
def mem(tmp_path):
    return Memory(StubLLM(), db_path=tmp_path / "test.db")


@pytest.fixture
def reg(mem, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "ws")
    config.WORKSPACE.mkdir()
    monkeypatch.setattr(config, "ALLOWED_DIRS", [tmp_path])
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_URL", "")
    # CI runners have no global git identity — exercise OSIRIS's own injection
    monkeypatch.setattr(config, "GIT_NAME", "osiris-test")
    monkeypatch.setattr(config, "GIT_EMAIL", "osiris@test.local")
    from osiris import google_services
    monkeypatch.setattr(google_services, "CREDENTIALS_FILE", tmp_path / "nope.json")
    monkeypatch.setattr(google_services, "TOKEN_FILE", tmp_path / "notoken.json")
    return build_registry(mem, Planner(StubLLM(), mem))


# -- memory ----------------------------------------------------------------

def test_memory_store_and_recall(mem):
    mem.remember("user targets IBM Quantum roles in 2027")
    hits = mem.recall("what career is the user targeting", min_score=0.0)
    assert hits and "IBM Quantum" in hits[0]


def test_recall_backfills_null_embeddings(mem):
    mem.db.execute("INSERT INTO memories (text, kind, embedding, created_at) VALUES (?,?,?,?)",
                   ("loves the terminal aesthetic", "fact", None, "2026-01-01"))
    mem.db.commit()
    hits = mem.recall("terminal aesthetic", top_k=5, min_score=0.0)
    assert any("terminal" in h for h in hits)
    assert mem.db.execute("SELECT embedding FROM memories").fetchone()[0] is not None


def test_recall_skips_mismatched_dimensions(mem):
    mem.db.execute("INSERT INTO memories (text, kind, embedding, created_at) VALUES (?,?,?,?)",
                   ("old-model memory", "fact", json.dumps([1.0] * 8), "2026-01-01"))
    mem.db.commit()
    assert mem.recall("anything", min_score=0.0) == []  # skipped, not garbage-scored


# -- planner / scheduler ----------------------------------------------------------------

def test_planner_parse_and_persist(mem):
    result = Planner(StubLLM(), mem).create_plan("Get into Stanford AI")
    assert len(result["milestones"]) == 2
    goals = mem.list_goals()
    assert goals[0]["milestones"][0]["title"] == "Learn linear algebra"
    assert mem.complete_milestone(goals[0]["milestones"][0]["id"])


def test_scheduler_lifecycle(mem):
    tid = mem.add_task("review threat model", "2020-01-01T09:00:00")
    assert mem.due_tasks()[0]["id"] == tid
    assert any(t["id"] == tid for t in mem.due_unnotified())
    mem.mark_notified(tid)
    assert not mem.due_unnotified()
    assert mem.complete_task(tid) and not mem.due_tasks()
    with pytest.raises(ValueError):
        mem.add_task("bad", "not-a-date")


# -- tools ----------------------------------------------------------------

def test_registry_complete_and_well_formed(reg):
    specs = reg.specs()
    assert len(specs) == 36
    assert all({"name", "description", "parameters"} <= set(s) for s in specs)
    assert "unknown tool" in reg.execute("nope", {})


def test_workspace_sandbox(reg):
    assert "wrote 5 chars" in reg.execute("write_file", {"path": "n/t.md", "content": "hello"})
    assert reg.execute("read_file", {"path": "n/t.md"}) == "hello"
    assert "escapes" in reg.execute("write_file", {"path": "../x", "content": "x"})
    # "~" must expand and be blocked, never become a literal directory in workspace
    assert "escapes" in reg.execute("write_file", {"path": "~/.env", "content": "SECRET"})
    assert not (config.WORKSPACE / "~").exists()


def test_fs_tilde_expands_and_is_guarded(reg):
    out = reg.execute("fs_read", {"path": "~/.env"})
    assert "outside allowed" in out or "FileNotFoundError" in out
    assert "literal" not in out


def test_fs_allowed_roots(reg, tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "a.txt").write_text("data", encoding="utf-8")
    assert reg.execute("fs_read", {"path": str(tmp_path / "proj" / "a.txt")}) == "data"
    assert "a.txt" in reg.execute("fs_list", {"path": str(tmp_path / "proj")})
    outside = tmp_path.parent / "outside.txt"  # sibling of the only allowed root
    assert "outside allowed" in reg.execute("fs_read", {"path": str(outside)})


def test_fs_write_and_edit_in_allowed_roots(reg, tmp_path):
    out = reg.execute("fs_write", {"path": str(tmp_path / "p" / "x.py"), "content": "a = 1\nb = 2\n"})
    assert "wrote" in out
    assert "replaced 1" in reg.execute(
        "fs_edit", {"path": str(tmp_path / "p" / "x.py"), "find": "a = 1", "replace": "a = 99"})
    assert "a = 99" in (tmp_path / "p" / "x.py").read_text()
    assert "not found" in reg.execute(
        "fs_edit", {"path": str(tmp_path / "p" / "x.py"), "find": "zzz", "replace": "y"})
    assert "outside allowed" in reg.execute(
        "fs_write", {"path": str(tmp_path.parent / "esc.py"), "content": "x"})


def test_git_list_repos_finds_repos(reg, tmp_path):
    (tmp_path / "myrepo" / ".git").mkdir(parents=True)
    (tmp_path / "notarepo").mkdir()
    out = reg.execute("git_list_repos", {})
    assert "myrepo" in out and "notarepo" not in out


def test_graceful_errors_without_credentials(reg):
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_issues", {"repo": "a/b"})
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_repos", {})
    assert "DISCORD_WEBHOOK_URL not set" in reg.execute("discord_send", {"message": "x"})
    assert "credentials.json missing" in reg.execute("gmail_search", {"query": "x"})
    assert "credentials.json missing" in reg.execute(
        "gmail_create_draft", {"to": "a@b.c", "subject": "s", "body": "b"})


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="git not installed")
def test_git_tools_on_real_repo(reg, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env_id = ["-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "f.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo)] + env_id + ["add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo)] + env_id + ["commit", "-qm", "init"], check=True)

    assert "init" in reg.execute("git_log", {"repo": str(repo), "limit": 1})
    (repo / "f.txt").write_text("v2", encoding="utf-8")
    sync = json.loads(reg.execute("git_sync_check", {"repo": str(repo)}))
    assert sync["in_sync"] is False and sync["uncommitted_files"]
    out = reg.execute("git_commit", {"repo": str(repo), "message": "update"})
    assert "update" in out
    sync = json.loads(reg.execute("git_sync_check", {"repo": str(repo)}))
    assert sync["uncommitted_files"] == []
    assert "outside allowed" in reg.execute("git_status", {"repo": str(tmp_path.parent / "elsewhere")})


# -- router ----------------------------------------------------------------

class DeadLLM(BaseLLM):
    name = "dead"
    def __init__(self):
        self.calls = 0
    def chat(self, *a, **k):
        self.calls += 1
        raise ConnectionError("429 rate limit boom")


def test_router_failover():
    r = ModelRouter(chain=[DeadLLM(), StubLLM()], verbose=False)
    assert "linear algebra" in r.chat([{"role": "user", "content": "x"}]).text
    assert r.describe() == "dead → stub"
    assert "stub" in r.active


def test_router_cooldown_skips_failed_provider():
    dead = DeadLLM()
    r = ModelRouter(chain=[dead, StubLLM()], verbose=False)
    r.chat([{"role": "user", "content": "a"}])
    assert dead.calls == 1
    r.chat([{"role": "user", "content": "b"}])  # dead is cooling down -> not retried
    assert dead.calls == 1


def test_router_all_dead_and_empty(monkeypatch):
    import osiris.router as router_mod
    monkeypatch.setattr(router_mod.time, "sleep", lambda s: None)
    dead = DeadLLM()
    with pytest.raises(RuntimeError, match="rate-limited or unreachable"):
        ModelRouter(chain=[dead], verbose=False).chat([{"role": "user", "content": "x"}])
    assert dead.calls == 2  # first pass + post-breather retry pass
    with pytest.raises(RuntimeError, match="no LLM providers"):
        ModelRouter(chain=[])


def test_embeddings_flag_off_uses_keyword_search(mem, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_ENABLED", False)
    mem.remember("user prefers quantum computing projects")
    row = mem.db.execute("SELECT embedding FROM memories").fetchone()
    assert row["embedding"] is None  # no API spend
    assert mem.recall("quantum projects") == ["user prefers quantum computing projects"]


# -- agents ----------------------------------------------------------------

def test_agent_routing():
    assert pick_agent("any unread emails today?").key == "email"
    assert pick_agent("commit and push my repo").key == "coding"
    assert pick_agent("plan my study roadmap").key == "learning"
    assert pick_agent("hello!").key == "personal"


def test_profiles_reference_real_tools(reg):
    names = {s["name"] for s in reg.specs()}
    for p in PROFILES.values():
        assert set(p.tools) <= names, p.key


def test_history_trim_keeps_turn_boundaries(mem, reg):
    agent = Agent(StubLLM(), mem, reg)
    for i in range(config.MAX_HISTORY + 20):
        agent.history.append({"role": "user" if i % 2 == 0 else "assistant",
                              "content": f"m{i}", "tool_calls": []})
    agent._trim_history()
    assert len(agent.history) <= config.MAX_HISTORY
    assert agent.history[0]["role"] == "user"


# -- reflection ----------------------------------------------------------------

class ReflectStub(BaseLLM):
    """Returns a fixed set of extracted facts."""
    name = "reflect-stub"

    def __init__(self, facts):
        self.facts = facts

    def chat(self, messages, system="", tools=None):
        return LLMResponse(text=json.dumps(self.facts))

    def embed(self, text):
        return StubLLM().embed(text)


def _history(n=4):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
            for i in range(n)]


def test_reflection_stores_new_facts(mem):
    from osiris.reflection import reflect
    llm = ReflectStub(["prefers dark terminal UIs", "targets IBM Quantum roles"])
    stored = reflect(llm, mem, _history())
    assert len(stored) == 2
    assert mem.recall("terminal UI preference", min_score=0.0)


def test_reflection_dedupes(mem):
    from osiris.reflection import reflect
    llm = ReflectStub(["prefers dark terminal UIs"])
    assert len(reflect(llm, mem, _history())) == 1
    assert reflect(llm, mem, _history()) == []  # second pass: exact dup skipped


def test_reflection_skips_trivial_sessions(mem):
    from osiris.reflection import reflect
    llm = ReflectStub(["should never be stored"])
    assert reflect(llm, mem, [{"role": "user", "content": "hi"}]) == []


def test_reflection_survives_malformed_output(mem):
    from osiris.reflection import reflect
    llm = ReflectStub("not json at all")
    llm.chat = lambda *a, **k: LLMResponse(text="no array here")
    assert reflect(llm, mem, _history()) == []


# -- provider message conversion ----------------------------------------------------------------

def test_openai_ids_match_for_idless_tool_calls():
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [ToolCall("a", {}, id=""), ToolCall("b", {}, id="")]},
        {"role": "tool", "name": "a", "tool_call_id": "", "content": "ra"},
        {"role": "tool", "name": "b", "tool_call_id": "", "content": "rb"},
    ]
    out = _to_openai_messages(msgs, "")
    ids = [c["id"] for c in out[0]["tool_calls"]]
    assert ids == [out[1]["tool_call_id"], out[2]["tool_call_id"]]
    assert out[0]["content"] == ""


def test_pseudo_function_calls_rescued_from_text():
    from osiris.providers.openai_compat import _rescue_pseudo_calls
    out = LLMResponse(text='Sure! <function=remember {"text": "user likes green"}</function> Done.')
    out = _rescue_pseudo_calls(out)
    assert out.tool_calls[0].name == "remember"
    assert out.tool_calls[0].args == {"text": "user likes green"}
    assert "<function" not in out.text and "Done." in out.text


def test_null_tool_args_do_not_crash(mem, reg):
    class NullArgsLLM(BaseLLM):
        name = "nullargs"
        calls = 0
        def chat(self, messages, system="", tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=[ToolCall(name="list_tasks", args=None)])
            return LLMResponse(text="done")
        def embed(self, text):
            return StubLLM().embed(text)

    agent = Agent(NullArgsLLM(), mem, reg)
    seen = []
    reply = agent.run("list my tasks", on_tool=lambda n, a: seen.append(a.items() is not None))
    assert reply == "done" and seen == [True]


def test_gemini_skips_empty_assistant_turn():
    contents = _to_gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "user", "content": "again"},
    ])
    assert len(contents) == 2 and all(c.parts for c in contents)
