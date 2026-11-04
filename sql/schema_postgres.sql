-- PostgreSQL schema; also created by Database.init_schema().
CREATE TABLE IF NOT EXISTS users(
        user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, token_sha256 TEXT UNIQUE NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1);

CREATE TABLE IF NOT EXISTS user_groups(
        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE, group_name TEXT NOT NULL,
        PRIMARY KEY(user_id, group_name));

CREATE TABLE IF NOT EXISTS sessions(
        session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        state_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS turns(
        turn_id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        user_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
        cited_chunk_ids TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, turn_id);

CREATE TABLE IF NOT EXISTS documents(
        document_id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL,
        document_family TEXT NOT NULL, version TEXT NOT NULL, effective_from TEXT, effective_to TEXT,
        access_groups_json TEXT NOT NULL, chunks INTEGER NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS audit_log(
        event_id BIGSERIAL PRIMARY KEY, user_id TEXT, event TEXT NOT NULL, detail_json TEXT NOT NULL, created_at TEXT NOT NULL);
