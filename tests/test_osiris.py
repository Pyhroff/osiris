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
    assert len(specs) == 32
    assert all({"name", "description", "parameters"} <= set(s) for s in specs)
    assert "unknown tool" in reg.execute("nope", {})


def test_workspace_sandbox(reg):
    assert "wrote 5 chars" in reg.execute("write_file", {"path": "n/t.md", "content": "hello"})
    assert reg.execute("read_file", {"path": "n/t.md"}) == "hello"
    assert "escapes" in reg.execute("write_file", {"path": "../x", "content": "x"})


def test_fs_allowed_roots(reg, tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "a.txt").write_text("data", encoding="utf-8")
    assert reg.execute("fs_read", {"path": str(tmp_path / "proj" / "a.txt")}) == "data"
    assert "a.txt" in reg.execute("fs_list", {"path": str(tmp_path / "proj")})
    assert "outside allowed" in reg.execute("fs_read", {"path": "C:/Windows/win.ini"})


def test_graceful_errors_without_credentials(reg):
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_issues", {"repo": "a/b"})
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
    assert "outside allowed" in reg.execute("git_status", {"repo": "C:/Windows"})


# -- router ----------------------------------------------------------------

class DeadLLM(BaseLLM):
    name = "dead"
    def chat(self, *a, **k):
        raise ConnectionError("boom")


def test_router_failover():
    r = ModelRouter(chain=[DeadLLM(), StubLLM()], verbose=False)
    assert "linear algebra" in r.chat([{"role": "user", "content": "x"}]).text
    assert r.describe() == "dead → stub"


def test_router_all_dead_and_empty():
    with pytest.raises(RuntimeError, match="all providers failed"):
        ModelRouter(chain=[DeadLLM()], verbose=False).chat([{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeError, match="no LLM providers"):
        ModelRouter(chain=[])


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


def test_gemini_skips_empty_assistant_turn():
    contents = _to_gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "user", "content": "again"},
    ])
    assert len(contents) == 2 and all(c.parts for c in contents)
