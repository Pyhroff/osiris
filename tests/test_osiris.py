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
    assert len(specs) == 59
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


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="git not installed")
def test_git_checkout_creates_and_switches_branch(reg, tmp_path):
    repo = tmp_path / "branchrepo"
    reg.execute("git_init", {"path": str(repo)})
    (repo / "f.txt").write_text("v1", encoding="utf-8")
    reg.execute("git_commit", {"repo": str(repo), "message": "init"})

    out = reg.execute("git_checkout", {"repo": str(repo), "branch": "feature-x"})
    assert "error" not in out.lower()
    current = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                             capture_output=True, text=True).stdout.strip()
    assert current == "feature-x"

    # switching to an existing branch (create=False) rather than making a new one
    out2 = reg.execute("git_checkout", {"repo": str(repo), "branch": "main", "create": False})
    assert "error" not in out2.lower()
    current2 = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                              capture_output=True, text=True).stdout.strip()
    assert current2 == "main"


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="git not installed")
def test_git_checkout_create_true_switches_to_already_existing_branch(reg, tmp_path):
    """create=True (the default) is documented as 'creating it first if it
    doesn't exist' — calling it a second time on the same branch (e.g. to add
    another commit before a PR) must switch, not error with 'already exists'."""
    repo = tmp_path / "branchrepo2"
    reg.execute("git_init", {"path": str(repo)})
    (repo / "f.txt").write_text("v1", encoding="utf-8")
    reg.execute("git_commit", {"repo": str(repo), "message": "init"})

    reg.execute("git_checkout", {"repo": str(repo), "branch": "feature-x"})
    reg.execute("git_checkout", {"repo": str(repo), "branch": "main", "create": False})

    out = reg.execute("git_checkout", {"repo": str(repo), "branch": "feature-x"})  # create=True default
    assert "error" not in out.lower() and "already exists" not in out.lower()
    current = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                             capture_output=True, text=True).stdout.strip()
    assert current == "feature-x"


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


# -- backup / export --------------------------------------------------------

def test_memory_export_all_shape(mem):
    mem.remember("favorite editor: neovim")
    mem.add_task("ship the thing", "2026-09-01T09:00:00")
    dump = mem.export_all()
    assert set(dump) == {"exported_at", "memories", "goals", "tasks", "applications"}
    assert any(m["text"] == "favorite editor: neovim" for m in dump["memories"])
    assert any(t["description"] == "ship the thing" for t in dump["tasks"])
    # embeddings are deliberately excluded from the backup — regenerable, bulky
    assert "embedding" not in dump["memories"][0]


def test_export_memory_tool_writes_backup_file(reg, mem):
    mem.remember("test fact for backup")
    out = reg.execute("export_memory", {})
    assert "backed up" in out and "1 memories" in out
    backups = list((config.WORKSPACE / "backups").glob("osiris-backup-*.json"))
    assert len(backups) == 1
    saved = json.loads(backups[0].read_text(encoding="utf-8"))
    assert any(m["text"] == "test fact for backup" for m in saved["memories"])


# -- /status live checks (main.py) -------------------------------------------

def test_check_github_reports_invalid_token(monkeypatch):
    import main
    monkeypatch.setattr(config, "GITHUB_TOKEN", "bad-token")
    class FakeResp:
        status_code = 401
        def json(self): return {}
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: FakeResp())
    assert "not working" in main._check_github()


def test_check_github_reports_valid_token(monkeypatch):
    import main
    monkeypatch.setattr(config, "GITHUB_TOKEN", "good-token")
    class FakeResp:
        status_code = 200
        def json(self): return {"login": "Pyhroff"}
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: FakeResp())
    assert "authorized as Pyhroff" in main._check_github()


def test_check_github_missing_token(monkeypatch):
    import main
    monkeypatch.setattr(config, "GITHUB_TOKEN", "")
    assert "not set" in main._check_github()


def test_check_gmail_not_yet_authorized(monkeypatch, tmp_path):
    import main
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "credentials.json").write_text("{}")
    assert "not yet authorized" in main._check_gmail()


def test_check_gmail_missing_credentials(monkeypatch, tmp_path):
    import main
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert "credentials.json missing" in main._check_gmail()


def test_check_ollama_not_running(monkeypatch):
    import main
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://localhost:1")  # nothing listens here
    assert "not running" in main._check_ollama()


def test_graceful_errors_without_credentials(reg):
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_issues", {"repo": "a/b"})
    assert "GITHUB_TOKEN not set" in reg.execute("github_list_repos", {})
    assert "DISCORD_WEBHOOK_URL not set" in reg.execute("discord_send", {"message": "x"})
    assert "credentials.json missing" in reg.execute("gmail_search", {"query": "x"})
    assert "credentials.json missing" in reg.execute(
        "gmail_create_draft", {"to": "a@b.c", "subject": "s", "body": "b"})
    assert "credentials.json missing" in reg.execute(
        "gmail_reply", {"message_id": "abc", "body": "hi"})


def test_clean_api_error_extracts_google_message():
    from osiris.google_services import _clean_api_error
    fake = Exception('<HttpError 403 when requesting ...>: {"error": '
                     '{"code": 403, "message": "Insufficient Permission"}}')
    assert _clean_api_error(fake) == "Insufficient Permission"
    unparseable = Exception("some other exception with no JSON body at all")
    assert _clean_api_error(unparseable) == str(unparseable)


def test_clean_errors_decorator_wraps_http_error_cleanly():
    from googleapiclient.errors import HttpError
    from osiris.google_services import _clean_errors
    import httplib2

    @_clean_errors
    def boom():
        resp = httplib2.Response({"status": "429"})
        raise HttpError(resp, b'{"error": {"message": "Quota exceeded"}}')

    try:
        boom()
        assert False, "should have raised"
    except RuntimeError as e:
        assert "Quota exceeded" in str(e)


def test_token_refresh_failure_gives_actionable_message(reg, monkeypatch, tmp_path):
    from osiris import google_services as g
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token": "x"}', encoding="utf-8")
    monkeypatch.setattr(g, "TOKEN_FILE", token_file)
    monkeypatch.setattr(g, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    g._services.clear()

    class DeadCreds:
        valid = False
        expired = True
        refresh_token = "r"
        def refresh(self, request):
            raise Exception("invalid_grant")

    monkeypatch.setattr(
        "google.oauth2.credentials.Credentials.from_authorized_user_file",
        lambda *a, **k: DeadCreds())

    try:
        g._get_service("gmail", "v1")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "delete token.json" in str(e) and "re-authorize" in str(e)


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


def test_context_overflow_gets_a_named_reason_not_generic_noise():
    from osiris.router import _reason
    # A big scaffold_project payload exceeding a smaller model's context window
    # used to surface as generic exception noise a user paraphrased as
    # "memory is reached, cannot proceed" — this names the real cause instead.
    e = Exception("Error code: 400 - This model's maximum context length is 8192 tokens")
    assert "too large for that model's context window" in _reason(e)
    e2 = Exception("400 input is too long for requested model")
    assert "too large for that model's context window" in _reason(e2)


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


# -- Ollama native adapter (think:false path) ---------------------------------------------------

def test_ollama_native_messages_pass_args_as_dict_not_json_string():
    """The one real wire-format difference from _to_openai_messages: native
    /api/chat wants tool_call arguments as an actual object, verified live
    — json.dumps()-ing them (the OpenAI-compat convention) would be wrong
    here, not just a style choice."""
    from osiris.providers.ollama_native import _to_native_messages
    msgs = [{"role": "assistant", "content": None,
            "tool_calls": [ToolCall("git_status", {"repo": "osiris"}, id="call_1")]}]
    out = _to_native_messages(msgs, "")
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"repo": "osiris"}


def test_ollama_native_ids_match_for_idless_tool_calls():
    from osiris.providers.ollama_native import _to_native_messages
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [ToolCall("a", {}, id=""), ToolCall("b", {}, id="")]},
        {"role": "tool", "name": "a", "tool_call_id": "", "content": "ra"},
        {"role": "tool", "name": "b", "tool_call_id": "", "content": "rb"},
    ]
    out = _to_native_messages(msgs, "")
    ids = [c["id"] for c in out[0]["tool_calls"]]
    assert ids == [out[1]["tool_call_id"], out[2]["tool_call_id"]]


def test_ollama_native_chat_parses_message_not_choices(monkeypatch):
    """The other real wire-format difference: native wraps the reply in
    "message" directly, not "choices"[0]["message"] like OpenAI-compat."""
    monkeypatch.setattr(config, "OLLAMA_THINK", False)  # explicit, not relying on the real .env
    from osiris.providers.ollama_native import OllamaNativeLLM
    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"message": {"role": "assistant", "content": "clean, up to date"}}
    captured = {}
    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()
    monkeypatch.setattr("osiris.providers.ollama_native.requests.post", fake_post)
    llm = OllamaNativeLLM("llama3.2", host="http://localhost:11434")
    out = llm.chat([{"role": "user", "content": "check status"}])
    assert out.text == "clean, up to date"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["think"] is False


def test_ollama_native_tool_call_args_already_a_dict(monkeypatch):
    """Confirms the adapter does NOT json.loads() dict-shaped arguments —
    the response parser must accept them as-is, matching what's actually
    observed on the wire, not assume a JSON string like OpenAI-compat."""
    from osiris.providers.ollama_native import OllamaNativeLLM
    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "git_status", "arguments": {"repo": "osiris"}}}
            ]}}
    monkeypatch.setattr("osiris.providers.ollama_native.requests.post", lambda *a, **k: FakeResp())
    llm = OllamaNativeLLM("llama3.2")
    out = llm.chat([{"role": "user", "content": "x"}])
    assert out.tool_calls[0].name == "git_status"
    assert out.tool_calls[0].args == {"repo": "osiris"}


def test_ollama_native_think_config_default_is_off(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_THINK", False)
    from osiris.providers.ollama_native import OllamaNativeLLM
    assert OllamaNativeLLM("llama3.2").think is False


def test_ollama_native_think_can_be_overridden_per_instance():
    from osiris.providers.ollama_native import OllamaNativeLLM
    assert OllamaNativeLLM("qwen3", think=True).think is True


def test_ollama_native_embed_not_implemented():
    from osiris.providers.ollama_native import OllamaNativeLLM
    with pytest.raises(NotImplementedError):
        OllamaNativeLLM("llama3.2").embed("text")


def test_make_ollama_uses_native_adapter_not_openai_compat(monkeypatch):
    """Regression guard for the actual point of this change — _make must
    route Ollama through OllamaNativeLLM, not silently fall back to the
    shared adapter that can't honor think:false."""
    from osiris.providers import _make
    monkeypatch.setattr("osiris.providers._ollama_model", lambda: "llama3.2:latest")
    llm = _make("ollama")
    from osiris.providers.ollama_native import OllamaNativeLLM
    assert isinstance(llm, OllamaNativeLLM)
    assert llm.model == "llama3.2:latest"


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


# -- discord bot: image attachments ------------------------------------------

def test_save_image_attachment_writes_file_with_safe_generated_name(tmp_path, monkeypatch):
    import discord_bot
    monkeypatch.setattr(discord_bot, "_IMAGE_INBOX", tmp_path / "inbox")
    saved = discord_bot._save_image_attachment(b"fake-png-bytes", "image/png", 0)
    assert saved is not None and saved.exists()
    assert saved.read_bytes() == b"fake-png-bytes"
    assert saved.suffix == ".png"
    assert saved.parent == tmp_path / "inbox"


def test_save_image_attachment_ignores_content_type_params(tmp_path, monkeypatch):
    import discord_bot
    monkeypatch.setattr(discord_bot, "_IMAGE_INBOX", tmp_path / "inbox")
    # discord.py sometimes reports content_type as "image/jpeg; charset=..."
    saved = discord_bot._save_image_attachment(b"x", "image/jpeg; charset=utf-8", 1)
    assert saved is not None and saved.suffix == ".jpg"


def test_save_image_attachment_rejects_unsupported_type(tmp_path, monkeypatch):
    import discord_bot
    monkeypatch.setattr(discord_bot, "_IMAGE_INBOX", tmp_path / "inbox")
    assert discord_bot._save_image_attachment(b"data", "application/pdf", 0) is None
    assert not (tmp_path / "inbox").exists()  # nothing written for a rejected type


def test_save_image_attachment_never_uses_caller_supplied_filename(tmp_path, monkeypatch):
    """The function signature has no filename parameter at all — verifies a
    crafted Discord filename (e.g. '../../evil.png') has no path to reach the
    save location, since it's never consulted in the first place."""
    import inspect
    import discord_bot
    assert "filename" not in inspect.signature(discord_bot._save_image_attachment).parameters


def test_build_content_with_images_appends_note_to_existing_text():
    import discord_bot
    from pathlib import Path
    out = discord_bot._build_content_with_images("what's this bug", [Path("/inbox/a.png")])
    assert out.startswith("what's this bug")
    assert "analyze_image" in out and "a.png" in out


def test_build_content_with_images_no_images_is_passthrough():
    import discord_bot
    assert discord_bot._build_content_with_images("hello", []) == "hello"


def test_build_content_with_images_empty_text_with_image_still_actionable():
    import discord_bot
    from pathlib import Path
    out = discord_bot._build_content_with_images("", [Path("/inbox/a.png")])
    assert out.strip()  # a bare photo with no caption must still produce non-empty content
    assert "analyze_image" in out


def test_build_content_with_images_multiple_images_all_referenced():
    import discord_bot
    from pathlib import Path
    out = discord_bot._build_content_with_images("compare these", [Path("/inbox/a.png"), Path("/inbox/b.png")])
    assert "a.png" in out and "b.png" in out


# -- schema_repair (constrained-decode argument repair) ----------------------

def test_needs_repair_true_when_required_key_missing():
    from osiris.providers import schema_repair
    schema = {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}
    assert schema_repair.needs_repair({}, schema) is True
    assert schema_repair.needs_repair({"repo": "osiris"}, schema) is False


def test_needs_repair_true_when_args_not_a_dict():
    from osiris.providers import schema_repair
    schema = {"type": "object", "properties": {}, "required": []}
    assert schema_repair.needs_repair(None, schema) is True
    assert schema_repair.needs_repair("not a dict", schema) is True


def test_needs_repair_false_for_no_required_fields():
    """A tool like current_datetime has no required args at all — empty
    args should never be flagged as needing repair."""
    from osiris.providers import schema_repair
    schema = {"type": "object", "properties": {}}  # no "required" key at all
    assert schema_repair.needs_repair({}, schema) is False


def test_repair_args_returns_none_on_network_failure(monkeypatch):
    from osiris.providers import schema_repair
    def boom(*a, **k):
        raise schema_repair.requests.RequestException("connection refused")
    monkeypatch.setattr(schema_repair.requests, "post", boom)
    result = schema_repair.repair_args(
        "http://localhost:11434/v1", "ollama", "llama3.2",
        "git_status", {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        "check status of osiris")
    assert result is None


def test_repair_args_returns_none_on_malformed_json_response(monkeypatch):
    from osiris.providers import schema_repair
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "not valid json{{"}}]}
    monkeypatch.setattr(schema_repair.requests, "post", lambda *a, **k: FakeResp())
    result = schema_repair.repair_args(
        "http://localhost:11434/v1", "ollama", "llama3.2",
        "git_status", {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        "check status of osiris")
    assert result is None


def test_repair_args_returns_parsed_dict_on_success(monkeypatch):
    from osiris.providers import schema_repair
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": '{"repo": "osiris"}'}}]}
    captured = {}
    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()
    monkeypatch.setattr(schema_repair.requests, "post", fake_post)
    result = schema_repair.repair_args(
        "http://localhost:11434/v1", "ollama", "llama3.2",
        "git_status", {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]},
        "check status of osiris")
    assert result == {"repo": "osiris"}
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["payload"]["response_format"]["json_schema"]["name"] == "git_status"
