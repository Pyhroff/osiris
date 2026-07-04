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

def test_memory_store_and_recall(mem, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_ENABLED", True)  # user .env may say off
    mem.remember("user targets IBM Quantum roles in 2027")
    hits = mem.recall("what career is the user targeting", min_score=0.0)
    assert hits and "IBM Quantum" in hits[0]


def test_forget_deletes_by_id(mem):
    mid = mem.remember("Cybot is a specialized cybersecurity OSIRIS agent")
    assert any(m["id"] == mid for m in mem.list_memories())
    assert mem.forget(mid) is True
    assert not any(m["id"] == mid for m in mem.list_memories())
    assert mem.forget(999999) is False  # already gone / never existed


def test_journal_capture_and_read(reg):
    assert "noted in journal" in reg.execute("note", {"text": "learned about JWT replay attacks"})
    reg.execute("note", {"text": "second note today"})
    out = reg.execute("read_journal", {})
    assert "JWT replay" in out and "second note" in out
    assert "no journal entries" in reg.execute("read_journal", {"date": "2000-01-01"})


def test_read_pdf_rejects_non_pdf_and_missing(reg, tmp_path):
    (tmp_path / "notes.txt").write_text("plain text", encoding="utf-8")
    assert "not a PDF" in reg.execute("read_pdf", {"path": str(tmp_path / "notes.txt")})
    out = reg.execute("read_pdf", {"path": str(tmp_path / "ghost.pdf")})
    assert "outside allowed" in out or "could not read" in out or "FileNotFound" in out


def test_forget_tool_requires_valid_id(reg):
    mem_id_out = reg.execute("remember", {"text": "temp fact to delete"})
    mid = int(mem_id_out.split("#")[1])
    listed = json.loads(reg.execute("list_memories", {}))
    assert any(m["id"] == mid for m in listed)
    assert reg.execute("forget", {"memory_id": mid}) == "forgotten"
    assert reg.execute("forget", {"memory_id": mid}) == "error: no such memory"


def test_recall_backfills_null_embeddings(mem, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_ENABLED", True)
    mem.db.execute("INSERT INTO memories (text, kind, embedding, created_at) VALUES (?,?,?,?)",
                   ("loves the terminal aesthetic", "fact", None, "2026-01-01"))
    mem.db.commit()
    hits = mem.recall("terminal aesthetic", top_k=5, min_score=0.0)
    assert any("terminal" in h for h in hits)
    assert mem.db.execute("SELECT embedding FROM memories").fetchone()[0] is not None


def test_recall_skips_mismatched_dimensions(mem, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_ENABLED", True)
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
    assert len(specs) == 57
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
    assert "uncommitted_changes" in out  # one call now reports status too


# -- vision ----------------------------------------------------------------

def test_analyze_image_requires_gemini_key(reg, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    out = reg.execute("analyze_image", {"path": "shot.png"})
    assert "GEMINI_API_KEY" in out


def test_analyze_image_rejects_unsupported_type(reg, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4")
    out = reg.execute("analyze_image", {"path": str(tmp_path / "doc.pdf")})
    assert "unsupported image type" in out


# -- local document index ----------------------------------------------------

def test_index_folder_and_search_docs(reg, mem, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_ENABLED", True)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "auth.py").write_text("def login(user, password):\n    check_credentials(user, password)\n", encoding="utf-8")
    (proj / "README.md").write_text("A demo project with nothing about login.\n", encoding="utf-8")
    (proj / "logo.png").write_bytes(b"\x89PNG")  # non-indexable, should be skipped

    out = reg.execute("index_folder", {"path": str(proj)})
    assert "indexed 2 file(s)" in out

    hits = mem.search_docs("login credentials", top_k=5, min_score=0.0)
    assert any("auth.py" in h["path"] for h in hits)

    # re-running skips unchanged files
    out2 = reg.execute("index_folder", {"path": str(proj)})
    assert "indexed 0 file(s), 2 already up to date" in out2


def test_search_docs_empty_before_indexing(mem):
    assert mem.search_docs("anything") == []


# -- PDF generation ----------------------------------------------------------

def test_write_pdf_creates_file(reg):
    out = reg.execute("write_pdf", {
        "path": "reports/weekly.pdf", "title": "Weekly Report",
        "content": "# Summary\nThings went well.\n\n- did X\n- did Y\n"})
    assert "wrote PDF" in out
    pdf_path = config.WORKSPACE / "reports" / "weekly.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_write_pdf_adds_extension_and_sandboxes(reg):
    out = reg.execute("write_pdf", {"path": "notes", "title": "T", "content": "body"})
    assert (config.WORKSPACE / "notes.pdf").exists()
    assert "escapes" in reg.execute("write_pdf", {"path": "../escape", "title": "T", "content": "x"})


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="git not installed")
def test_git_init_creates_new_repo(reg, tmp_path):
    out = reg.execute("git_init", {"path": str(tmp_path / "fresh")})
    assert "error" not in out.lower() or "initialized" in out.lower()
    assert (tmp_path / "fresh" / ".git").exists()
    assert (tmp_path / "fresh" / ".gitignore").exists()  # secrets guard
    assert ".env" in (tmp_path / "fresh" / ".gitignore").read_text()
    assert "already a git repository" in reg.execute("git_init", {"path": str(tmp_path / "fresh")})
    assert "outside allowed" in reg.execute("git_init", {"path": str(tmp_path.parent / "evil")})


def test_scaffold_project_writes_whole_tree(reg, tmp_path):
    files = [{"path": "ARCHITECTURE.md", "content": "# design"},
             {"path": "src/app.py", "content": "print('hi')"},
             {"path": "tests/test_app.py", "content": "def test(): pass"}]
    out = reg.execute("scaffold_project", {"folder": str(tmp_path / "proj"), "files": files})
    assert "scaffolded 3 files" in out
    assert (tmp_path / "proj" / "src" / "app.py").read_text() == "print('hi')"
    out = reg.execute("scaffold_project",
                      {"folder": str(tmp_path / "proj"),
                       "files": [{"path": "../evil.py", "content": "x"}]})
    assert "escapes project folder" in out
    assert "non-empty array" in reg.execute("scaffold_project",
                                            {"folder": str(tmp_path / "p2"), "files": []})


def test_github_create_repo_requires_local_repo(reg, tmp_path):
    (tmp_path / "notrepo").mkdir()
    out = reg.execute("github_create_repo", {"repo": str(tmp_path / "notrepo")})
    assert "not a git repository" in out


def test_run_recon_blocks_unauthorized_targets(reg, monkeypatch):
    monkeypatch.setattr(config, "AUTHORIZED_TARGETS", "")
    out = reg.execute("run_recon", {"tool": "nmap", "target": "8.8.8.8"})  # public IP, not private/authorized
    assert "not localhost/private" in out and "OSIRIS_AUTHORIZED_TARGETS" in out


def test_run_recon_allows_localhost_and_private_ips(reg):
    out = reg.execute("run_recon", {"tool": "ping", "target": "127.0.0.1", "flags": ["-n"]})
    assert "not localhost/private" not in out  # got far enough to actually try running it


def test_run_recon_rejects_unknown_tool(reg):
    out = reg.execute("run_recon", {"tool": "sqlmap", "target": "127.0.0.1"})
    assert "not in allowlist" in out


def test_run_recon_rejects_unsafe_flags(reg, monkeypatch):
    monkeypatch.setattr(config, "AUTHORIZED_TARGETS", "lab.internal")
    out = reg.execute("run_recon", {"tool": "nmap", "target": "lab.internal",
                                    "flags": ["--script", "vuln"]})
    assert "not allowed" in out


def test_run_recon_honors_authorized_targets_env(reg, monkeypatch):
    monkeypatch.setattr(config, "AUTHORIZED_TARGETS", "lab.internal, other.test")
    out = reg.execute("run_recon", {"tool": "whois", "target": "lab.internal"})
    assert "not localhost/private" not in out


@pytest.mark.skipif(subprocess.run(["pytest", "--version"], capture_output=True).returncode != 0,
                    reason="pytest not on PATH")
def test_run_project_command_executes_pytest(reg, tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "test_ok.py").write_text("def test_pass():\n    assert True\n", encoding="utf-8")
    out = reg.execute("run_project_command", {"repo": str(tmp_path / "proj"), "command": "pytest"})
    assert "exit code 0" in out and "1 passed" in out


def test_run_project_command_rejects_unknown_command(reg, tmp_path):
    out = reg.execute("run_project_command", {"repo": str(tmp_path), "command": "rm-rf"})
    assert "not in allowlist" in out


def test_run_project_command_rejects_unsafe_extra_args(reg, tmp_path):
    out = reg.execute("run_project_command",
                      {"repo": str(tmp_path), "command": "pytest", "extra_args": ["; rm -rf /"]})
    assert "disallowed characters" in out


def test_cve_lookup_by_id(reg, monkeypatch):
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"vulnerabilities": [{"cve": {
                "id": "CVE-2021-44228", "published": "2021-12-10T10:00:00.000",
                "descriptions": [{"lang": "en", "value": "Log4j RCE via JNDI lookup."}],
                "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 10.0}}]},
            }}]}
    import osiris.tools as tools_mod
    monkeypatch.setattr(tools_mod.requests, "get", lambda *a, **k: FakeResp())
    out = json.loads(reg.execute("cve_lookup", {"query": "CVE-2021-44228"}))
    assert out[0]["id"] == "CVE-2021-44228" and out[0]["cvss"] == 10.0


def test_cve_lookup_network_failure_is_graceful(reg, monkeypatch):
    import osiris.tools as tools_mod
    def boom(*a, **k):
        raise tools_mod.requests.RequestException("timeout")
    monkeypatch.setattr(tools_mod.requests, "get", boom)
    out = reg.execute("cve_lookup", {"query": "log4j"})
    assert "NVD lookup failed" in out


def test_web_search_requires_api_key(reg, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    out = reg.execute("web_search", {"query": "fifa world cup 2026 schedule"})
    assert "TAVILY_API_KEY not set" in out


def test_web_search_returns_parsed_results(reg, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key")
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"results": [{"title": "FIFA 2026", "url": "https://fifa.com/2026",
                                 "content": "The 2026 World Cup schedule."}]}
    import osiris.tools as tools_mod
    monkeypatch.setattr(tools_mod.requests, "post", lambda *a, **k: FakeResp())
    out = json.loads(reg.execute("web_search", {"query": "fifa world cup 2026"}))
    assert out[0]["title"] == "FIFA 2026" and out[0]["url"] == "https://fifa.com/2026"


def test_application_tracking_lifecycle(reg):
    out = reg.execute("track_application",
                      {"company": "Cyberops", "role": "VAPT Intern", "contact_email": "hr@cyberops.test"})
    assert "logged application #" in out
    app_id = int(out.split("#")[1])
    apps = json.loads(reg.execute("list_applications", {}))
    assert any(a["id"] == app_id and a["company"] == "Cyberops" for a in apps)
    assert reg.execute("update_application_status", {"application_id": app_id, "status": "interview"}) == "updated"
    apps = json.loads(reg.execute("list_applications", {"status": "interview"}))
    assert apps[0]["status"] == "interview"
    assert reg.execute("update_application_status", {"application_id": 99999, "status": "x"}) == "error: no such application"


def test_graceful_errors_without_credentials(reg):
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_issues", {"repo": "a/b"})
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_repos", {})
    assert "DISCORD_WEBHOOK_URL not set" in reg.execute("discord_send", {"message": "x"})
    assert "credentials.json missing" in reg.execute("gmail_search", {"query": "x"})
    assert "credentials.json missing" in reg.execute(
        "gmail_create_draft", {"to": "a@b.c", "subject": "s", "body": "b"})
    assert "credentials.json missing" in reg.execute(
        "gmail_reply", {"message_id": "abc", "body": "hi"})


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
    # architect must win even when github/commit appear alongside a build keyword
    assert pick_agent("build me a FastAPI REST API with tests").key == "architect"
    assert pick_agent("build me a react app and push to github").key == "architect"
    assert pick_agent("design a CLI tool with github integration").key == "architect"
    # domain security agents win over architect's broad "design a" on cyber-specific phrasing
    assert pick_agent("help me plan a bug hunt for this pentest").key == "redteam"
    assert pick_agent("draft an incident response playbook").key == "blueteam"
    assert pick_agent("design an adversarial jailbreak test suite for my llm").key == "aisec"


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


# -- daemon routines ----------------------------------------------------------------

def test_routine_due_logic():
    from datetime import datetime
    from daemon import routine_due
    r = {"name": "brief", "time": "08:30", "prompt": "x"}
    morning = datetime(2026, 7, 3, 9, 0)
    assert routine_due(r, "", morning)                            # never ran, time passed
    assert not routine_due(r, "2026-07-03T08:31:00", morning)     # already ran today
    assert routine_due(r, "2026-07-02T08:31:00", morning)         # ran yesterday
    assert not routine_due(r, "", datetime(2026, 7, 3, 8, 0))     # too early


def test_routine_due_weekly_cadence():
    from datetime import datetime
    from daemon import routine_due
    weekly = {"name": "digest", "time": "09:00", "days": ["Sun"], "prompt": "x"}
    assert not routine_due(weekly, "", datetime(2026, 7, 3, 10, 0))   # Friday, not Sunday
    assert routine_due(weekly, "", datetime(2026, 7, 5, 10, 0))       # Sunday, time passed
    assert not routine_due(weekly, "2026-07-05T09:01:00", datetime(2026, 7, 5, 10, 0))  # already ran


def test_routine_run_marking(mem):
    assert mem.routine_last_run("brief") == ""
    mem.mark_routine_run("brief")
    assert mem.routine_last_run("brief").startswith("20")
    mem.mark_routine_run("brief")  # upsert, no crash


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


def test_ollama_model_ranking():
    from osiris.providers import _pick_ollama
    assert _pick_ollama(["gemma3:4b", "llama3.2:latest", "qwen3:8b"]) == "qwen3:8b"
    assert _pick_ollama(["gemma3:4b", "llama3.2:latest"]) == "llama3.2:latest"
    assert _pick_ollama(["gemma3:4b"]) == "gemma3:4b"  # last resort, better than nothing
    assert _pick_ollama([]) is None


def test_extra_providers_join_chain_when_keyed(monkeypatch):
    from osiris.providers import build_chain
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(config, "CEREBRAS_API_KEY", "k")
    monkeypatch.setattr(config, "MISTRAL_API_KEY", "k")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(config, "GITHUB_MODELS_TOKEN", "k")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "PROVIDER_CHAIN", "gemini,groq,cerebras,mistral,openrouter,github,openai")
    assert [p.name for p in build_chain()] == ["groq", "cerebras", "mistral", "openrouter", "github"]


def test_bare_json_tool_call_rescued():
    from osiris.providers.openai_compat import _rescue_json_call
    out = LLMResponse(text='{"name": "recall", "parameters": {"query": "user prefs"}}')
    out = _rescue_json_call(out)
    assert out.tool_calls[0].name == "recall" and out.tool_calls[0].args == {"query": "user prefs"}
    assert out.text == ""
    ok = _rescue_json_call(LLMResponse(text="just a normal { curly } answer"))
    assert ok.text and not ok.tool_calls  # non-JSON text untouched


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


def test_markdown_stripped_from_replies(mem, reg):
    """No OSIRIS interface renders markdown — a provider that ignores the
    plain-text instruction (Groq's oss-120b does this in practice) must still
    not leak literal ** or # to the user."""
    class MarkdownLLM(BaseLLM):
        name = "markdown"
        def chat(self, messages, system="", tools=None):
            return LLMResponse(text="# Profile\n**Name:** Praising\n__Role:__ student\n*Goal*: SOC analyst")
        def embed(self, text):
            return StubLLM().embed(text)

    agent = Agent(MarkdownLLM(), mem, reg)
    reply = agent.run("tell me about myself")
    assert "**" not in reply and "__" not in reply and not reply.startswith("#")
    assert "*" not in reply
    assert "Name: Praising" in reply and "Role: student" in reply and "Goal: SOC analyst" in reply


def test_circuit_breaker_interrupts_repeated_tool_thrashing(mem, reg):
    """A weak model that keeps retrying the same failing tool call (observed live
    with Mistral thrashing on run_project_command) should get a corrective nudge
    injected into history rather than silently burning the whole step budget."""
    class ThrashingLLM(BaseLLM):
        name = "thrash"
        def chat(self, messages, system="", tools=None):
            # Bail out once the nudge shows up in history — proves it was injected.
            if any(m.get("role") == "user" and "Stop retrying the same way" in m.get("content", "")
                   for m in messages):
                return LLMResponse(text="done")
            return LLMResponse(tool_calls=[ToolCall(name="list_tasks", args={})])
        def embed(self, text):
            return StubLLM().embed(text)

    agent = Agent(ThrashingLLM(), mem, reg)
    reply = agent.run("do something that keeps failing")
    assert reply == "done"
    assert any("Stop retrying the same way" in (m.get("content") or "") for m in agent.history)


def test_empty_response_retries_instead_of_surrendering(mem, reg):
    """Observed live: Gemini returned zero text AND zero tool calls for a casual
    mail request, twice in a row (local REPL and the Discord bot). The old code
    gave up the whole turn on the first empty response — it should nudge and
    retry with the remaining step budget instead."""
    class FlakyLLM(BaseLLM):
        name = "flaky"
        calls = 0
        def chat(self, messages, system="", tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(text="", tool_calls=[])  # the observed failure mode
            return LLMResponse(text="here are your emails")
        def embed(self, text):
            return StubLLM().embed(text)

    llm = FlakyLLM()
    agent = Agent(llm, mem, reg)
    reply = agent.run("gimme my mail drtails")
    assert reply == "here are your emails"
    assert llm.calls == 2
    assert any("produced no reply" in (m.get("content") or "") for m in agent.history)


def test_repeated_empty_responses_eventually_hit_step_limit(mem, reg):
    """If the model keeps returning nothing at all, it must not loop forever —
    the existing step-limit message should still fire, just later."""
    class AlwaysEmptyLLM(BaseLLM):
        name = "empty"
        def chat(self, messages, system="", tools=None):
            return LLMResponse(text="", tool_calls=[])
        def embed(self, text):
            return StubLLM().embed(text)

    agent = Agent(AlwaysEmptyLLM(), mem, reg)
    reply = agent.run("anything")
    assert "stopped: hit the step limit" in reply


def test_gemini_skips_empty_assistant_turn():
    contents = _to_gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "user", "content": "again"},
    ])
    assert len(contents) == 2 and all(c.parts for c in contents)
