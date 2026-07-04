"""Persistent memory: facts (vector-searchable), goals/milestones, scheduled tasks.

One SQLite file, embeddings stored inline as JSON. Cosine search in Python is
plenty at personal scale (thousands of memories); swap in a real vector DB
only if this ever becomes the bottleneck.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime

from . import config
from .providers import BaseLLM

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY, text TEXT NOT NULL, kind TEXT DEFAULT 'fact',
    embedding TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY, goal_id INTEGER NOT NULL REFERENCES goals(id),
    title TEXT NOT NULL, detail TEXT, due TEXT, status TEXT DEFAULT 'pending',
    position INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY, description TEXT NOT NULL, due TEXT NOT NULL,
    done INTEGER DEFAULT 0, notified INTEGER DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS routine_runs (
    name TEXT PRIMARY KEY, last_run TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY, company TEXT NOT NULL, role TEXT, contact_email TEXT,
    status TEXT DEFAULT 'applied', applied_date TEXT NOT NULL, notes TEXT,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS doc_chunks (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL, embedding TEXT, mtime REAL NOT NULL, indexed_at TEXT NOT NULL,
    UNIQUE(path, chunk_index));
"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class Memory:
    def __init__(self, llm: BaseLLM, db_path=None):
        self.llm = llm
        # check_same_thread=False: the web server runs handlers in a threadpool
        self.db = sqlite3.connect(db_path or config.DB_PATH, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        try:  # migrate pre-daemon databases
            self.db.execute("ALTER TABLE tasks ADD COLUMN notified INTEGER DEFAULT 0")
            self.db.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    # -- facts ---------------------------------------------------------------
    def remember(self, text: str, kind: str = "fact") -> int:
        emb = None
        if config.EMBEDDINGS_ENABLED:
            try:
                emb = json.dumps(self.llm.embed(text))
            except Exception:
                emb = None  # degrade gracefully; keyword fallback still finds it
        cur = self.db.execute(
            "INSERT INTO memories (text, kind, embedding, created_at) VALUES (?,?,?,?)",
            (text, kind, emb, datetime.now().isoformat(timespec="seconds")))
        self.db.commit()
        return cur.lastrowid

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.35) -> list[str]:
        rows = self.db.execute("SELECT id, text, embedding FROM memories").fetchall()
        if not rows:
            return []
        if not config.EMBEDDINGS_ENABLED:
            return self._keyword_search(query, rows, top_k)
        try:
            q = self.llm.embed(query)
        except Exception:
            return self._keyword_search(query, rows, top_k)

        rows = self._backfill_embeddings(rows)
        scored = []
        for r in rows:
            if not r["embedding"]:
                continue
            vec = json.loads(r["embedding"])
            if len(vec) == len(q):  # guard against a changed embedding model
                scored.append((_cosine(q, vec), r["text"]))
        scored.sort(reverse=True)
        return [t for s, t in scored[:top_k] if s >= min_score]

    @staticmethod
    def _keyword_search(query: str, rows, top_k: int) -> list[str]:
        words = [w for w in query.lower().split() if len(w) > 3]
        return [r["text"] for r in rows
                if any(w in r["text"].lower() for w in words)][:top_k]

    def list_memories(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, text, kind, created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def forget(self, memory_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def _backfill_embeddings(self, rows, limit: int = 5):
        """Embed memories saved while embeddings were unavailable, a few per recall."""
        missing = [r for r in rows if not r["embedding"]][:limit]
        for r in missing:
            try:
                emb = json.dumps(self.llm.embed(r["text"]))
            except Exception:
                break
            self.db.execute("UPDATE memories SET embedding=? WHERE id=?", (emb, r["id"]))
        if missing:
            self.db.commit()
            return self.db.execute("SELECT id, text, embedding FROM memories").fetchall()
        return rows

    # -- goals & milestones ----------------------------------------------------
    def add_goal(self, title: str, milestones: list[dict]) -> int:
        cur = self.db.execute("INSERT INTO goals (title, created_at) VALUES (?,?)",
                              (title, datetime.now().isoformat(timespec="seconds")))
        goal_id = cur.lastrowid
        for i, m in enumerate(milestones):
            self.db.execute(
                "INSERT INTO milestones (goal_id, title, detail, due, position) VALUES (?,?,?,?,?)",
                (goal_id, m["title"], m.get("detail", ""), m.get("due"), i))
        self.db.commit()
        return goal_id

    def list_goals(self) -> list[dict]:
        out = []
        for g in self.db.execute("SELECT * FROM goals WHERE status='active'").fetchall():
            ms = self.db.execute(
                "SELECT * FROM milestones WHERE goal_id=? ORDER BY position", (g["id"],)).fetchall()
            out.append({"id": g["id"], "title": g["title"],
                        "milestones": [dict(m) for m in ms]})
        return out

    def complete_milestone(self, milestone_id: int) -> bool:
        cur = self.db.execute("UPDATE milestones SET status='done' WHERE id=?", (milestone_id,))
        self.db.commit()
        return cur.rowcount > 0

    # -- scheduled tasks -------------------------------------------------------
    def add_task(self, description: str, due_iso: str) -> int:
        datetime.fromisoformat(due_iso)  # validate early, fail loudly
        cur = self.db.execute(
            "INSERT INTO tasks (description, due, created_at) VALUES (?,?,?)",
            (description, due_iso, datetime.now().isoformat(timespec="seconds")))
        self.db.commit()
        return cur.lastrowid

    def list_tasks(self, include_done: bool = False) -> list[dict]:
        q = "SELECT * FROM tasks" + ("" if include_done else " WHERE done=0") + " ORDER BY due"
        return [dict(r) for r in self.db.execute(q).fetchall()]

    def due_tasks(self) -> list[dict]:
        now = datetime.now().isoformat(timespec="seconds")
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM tasks WHERE done=0 AND due<=? ORDER BY due", (now,)).fetchall()]

    def due_unnotified(self) -> list[dict]:
        now = datetime.now().isoformat(timespec="seconds")
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM tasks WHERE done=0 AND notified=0 AND due<=? ORDER BY due", (now,)).fetchall()]

    def mark_notified(self, task_id: int):
        self.db.execute("UPDATE tasks SET notified=1 WHERE id=?", (task_id,))
        self.db.commit()

    # -- proactive routines ------------------------------------------------------
    def routine_last_run(self, name: str) -> str:
        row = self.db.execute("SELECT last_run FROM routine_runs WHERE name=?", (name,)).fetchone()
        return row["last_run"] if row else ""

    def mark_routine_run(self, name: str):
        self.db.execute(
            "INSERT INTO routine_runs (name, last_run) VALUES (?,?) "
            "ON CONFLICT(name) DO UPDATE SET last_run=excluded.last_run",
            (name, datetime.now().isoformat(timespec="seconds")))
        self.db.commit()

    def complete_task(self, task_id: int) -> bool:
        cur = self.db.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
        self.db.commit()
        return cur.rowcount > 0

    # -- job/internship applications --------------------------------------------
    def add_application(self, company: str, role: str = "", contact_email: str = "",
                        notes: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO applications (company, role, contact_email, applied_date, notes, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (company, role, contact_email, datetime.now().date().isoformat(), notes,
             datetime.now().isoformat(timespec="seconds")))
        self.db.commit()
        return cur.lastrowid

    def list_applications(self, status: str = "") -> list[dict]:
        q = "SELECT * FROM applications" + (" WHERE status=?" if status else "") + " ORDER BY applied_date DESC"
        params = (status,) if status else ()
        return [dict(r) for r in self.db.execute(q, params).fetchall()]

    def update_application_status(self, application_id: int, status: str) -> bool:
        cur = self.db.execute("UPDATE applications SET status=? WHERE id=?", (status, application_id))
        self.db.commit()
        return cur.rowcount > 0

    # -- local document index (fs_* is on-demand reads; this is the searchable index) ---
    def doc_up_to_date(self, path: str, mtime: float) -> bool:
        row = self.db.execute("SELECT mtime FROM doc_chunks WHERE path=? LIMIT 1", (path,)).fetchone()
        return bool(row) and row["mtime"] >= mtime

    def index_chunk(self, path: str, chunk_index: int, text: str, mtime: float):
        emb = None
        if config.EMBEDDINGS_ENABLED:
            try:
                emb = json.dumps(self.llm.embed(text))
            except Exception:
                emb = None  # degrade gracefully; keyword fallback still finds it
        self.db.execute(
            "INSERT INTO doc_chunks (path, chunk_index, text, embedding, mtime, indexed_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(path, chunk_index) DO UPDATE SET "
            "text=excluded.text, embedding=excluded.embedding, mtime=excluded.mtime, "
            "indexed_at=excluded.indexed_at",
            (path, chunk_index, text, emb, mtime, datetime.now().isoformat(timespec="seconds")))
        self.db.commit()

    def search_docs(self, query: str, top_k: int = 5, min_score: float = 0.3) -> list[dict]:
        rows = self.db.execute("SELECT path, text, embedding FROM doc_chunks").fetchall()
        if not rows:
            return []
        if not config.EMBEDDINGS_ENABLED:
            words = [w for w in query.lower().split() if len(w) > 3]
            seen = {}
            for r in rows:
                if r["path"] not in seen and any(w in r["text"].lower() for w in words):
                    seen[r["path"]] = r["text"]
            return [{"path": p, "snippet": t[:300]} for p, t in list(seen.items())[:top_k]]
        try:
            q = self.llm.embed(query)
        except Exception:
            return []
        scored = []
        for r in rows:
            if not r["embedding"]:
                continue
            vec = json.loads(r["embedding"])
            if len(vec) == len(q):
                scored.append((_cosine(q, vec), r["path"], r["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_per_path: dict[str, tuple] = {}
        for score, path, text in scored:
            if score < min_score:
                break
            if path not in best_per_path:
                best_per_path[path] = (score, text)
            if len(best_per_path) >= top_k:
                break
        return [{"path": p, "score": round(s, 3), "snippet": t[:300]} for p, (s, t) in best_per_path.items()]
