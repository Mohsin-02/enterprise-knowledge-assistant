"""Relational store: users, group membership, chat sessions, document records, audit log.

SQLite by default; PostgreSQL when the URL starts with postgresql:// (needs `psycopg`). The table
layout matches the database file created earlier by `manage.py`, so existing data keeps working.
Only SHA-256 token hashes are stored; a token is shown once, when the user is created.
Rebuilt after an accidental overwrite of the original; the PostgreSQL branch is unexecuted here.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sqlite3
import threading

from .access import Principal

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users(
        user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, token_sha256 TEXT UNIQUE NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1)""",
    """CREATE TABLE IF NOT EXISTS user_groups(
        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE, group_name TEXT NOT NULL,
        PRIMARY KEY(user_id, group_name))""",
    """CREATE TABLE IF NOT EXISTS sessions(
        session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS turns(
        turn_id {pk}, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        user_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
        cited_chunk_ids TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, turn_id)",
    """CREATE TABLE IF NOT EXISTS documents(
        document_id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL,
        document_family TEXT NOT NULL, version TEXT NOT NULL, effective_from TEXT, effective_to TEXT,
        access_groups_json TEXT NOT NULL, chunks INTEGER NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS audit_log(
        event_id {pk}, user_id TEXT, event TEXT NOT NULL, detail_json TEXT NOT NULL, created_at TEXT NOT NULL)""",
]


def hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, url: str):
        self.url = url
        self.postgres = url.startswith(("postgres://", "postgresql://"))
        self._lock = threading.RLock()
        self._memory = None
        if not self.postgres:
            self.path = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url
            if self.path == ":memory:":
                self._memory = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self):
        if self.postgres:
            import psycopg  # optional dependency
            with psycopg.connect(self.url) as conn:
                yield conn
            return
        conn = self._memory or sqlite3.connect(self.path, timeout=30)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._memory is None:
                conn.close()

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.postgres else sql

    def run(self, sql: str, params: tuple = ()) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(self._sql(sql), params)

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock, self._connection() as conn:
            return list(conn.execute(self._sql(sql), params).fetchall())

    def init_schema(self) -> None:
        pk = "BIGSERIAL PRIMARY KEY" if self.postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        for statement in SCHEMA:
            self.run(statement.format(pk=pk))

    # ---- users ---------------------------------------------------------------------------------
    def upsert_user(self, user_id: str, display_name: str, groups: list[str], *, is_admin: bool = False) -> str | None:
        """Create or update a user. Returns a NEW token only when the user is created."""
        exists = self.query("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        token = None
        if exists:
            self.run("UPDATE users SET display_name = ?, is_admin = ?, active = 1 WHERE user_id = ?",
                     (display_name, int(is_admin), user_id))
        else:
            token = secrets.token_urlsafe(32)
            self.run("INSERT INTO users (user_id, display_name, token_sha256, is_admin, active) VALUES (?, ?, ?, ?, 1)",
                     (user_id, display_name, hash_token(token), int(is_admin)))
        self.set_groups(user_id, groups)
        return token

    def set_groups(self, user_id: str, groups: list[str]) -> None:
        self.run("DELETE FROM user_groups WHERE user_id = ?", (user_id,))
        for group in sorted({g.strip() for g in groups if g.strip()}):
            self.run("INSERT INTO user_groups (user_id, group_name) VALUES (?, ?)", (user_id, group))

    def set_active(self, user_id: str, active: bool) -> None:
        self.run("UPDATE users SET active = ? WHERE user_id = ?", (int(active), user_id))

    def authenticate(self, token: str | None) -> Principal | None:
        if not token:
            return None
        rows = self.query("SELECT user_id, is_admin FROM users WHERE token_sha256 = ? AND active = 1", (hash_token(token),))
        if not rows:
            return None
        user_id, is_admin = rows[0]
        groups = {g for (g,) in self.query("SELECT group_name FROM user_groups WHERE user_id = ?", (user_id,))}
        return Principal(user_id, frozenset(groups), bool(is_admin))

    # ---- sessions and turns --------------------------------------------------------------------
    def session_exists_for(self, session_id: str, user_id: str) -> bool:
        return bool(self.query("SELECT 1 FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id)))

    def save_session(self, session_id: str, user_id: str, state: dict) -> None:
        stamp = _now()
        self.run("INSERT INTO sessions (session_id, user_id, state_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?) "
                 "ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at "
                 "WHERE sessions.user_id = excluded.user_id", (session_id, user_id, json.dumps(state), stamp, stamp))

    def load_session(self, session_id: str, user_id: str) -> dict | None:
        rows = self.query("SELECT state_json FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        return json.loads(rows[0][0]) if rows else None

    def add_turn(self, session_id: str, user_id: str, role: str, content: str, cited_chunk_ids: list[str] | None = None) -> None:
        self.run("INSERT INTO turns (session_id, user_id, role, content, cited_chunk_ids, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                 (session_id, user_id, role, content, json.dumps(cited_chunk_ids or []), _now()))

    def turns(self, session_id: str, user_id: str) -> list[dict]:
        rows = self.query("SELECT role, content, cited_chunk_ids, created_at FROM turns "
                          "WHERE session_id = ? AND user_id = ? ORDER BY turn_id", (session_id, user_id))
        return [{"role": r, "content": c, "cited_chunk_ids": json.loads(ids), "created_at": t} for r, c, ids, t in rows]

    def delete_session(self, session_id: str, user_id: str) -> bool:
        if not self.session_exists_for(session_id, user_id):
            return False
        self.run("DELETE FROM turns WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        self.run("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        return True

    # ---- document records (a queryable mirror of the index) --------------------------------------
    def replace_documents(self, documents: list[dict]) -> None:
        keep = {d["document_id"] for d in documents}
        for (document_id,) in self.query("SELECT document_id FROM documents"):
            if document_id not in keep:
                self.run("DELETE FROM documents WHERE document_id = ?", (document_id,))
        for d in documents:
            self.run("INSERT INTO documents (document_id, source, title, document_family, version, effective_from, "
                     "effective_to, access_groups_json, chunks, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                     "ON CONFLICT(document_id) DO UPDATE SET source = excluded.source, title = excluded.title, "
                     "document_family = excluded.document_family, version = excluded.version, "
                     "effective_from = excluded.effective_from, effective_to = excluded.effective_to, "
                     "access_groups_json = excluded.access_groups_json, chunks = excluded.chunks, "
                     "updated_at = excluded.updated_at",
                     (d["document_id"], d["source"], d["title"], d["document_family"], d["version"],
                      d.get("effective_from"), d.get("effective_to"), json.dumps(d["access_groups"]),
                      d["chunks"], _now()))

    # ---- audit ---------------------------------------------------------------------------------
    def audit(self, user_id: str | None, event: str, detail: dict) -> None:
        self.run("INSERT INTO audit_log (user_id, event, detail_json, created_at) VALUES (?, ?, ?, ?)",
                 (user_id, event, json.dumps(detail, default=str), _now()))

    def audit_events(self, limit: int = 50) -> list[dict]:
        rows = self.query("SELECT user_id, event, detail_json, created_at FROM audit_log ORDER BY event_id DESC LIMIT ?", (limit,))
        return [{"user_id": u, "event": e, "detail": json.loads(d), "created_at": t} for u, e, d, t in rows]
