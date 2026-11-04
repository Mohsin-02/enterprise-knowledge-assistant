"""Application service used by api.py and manage.py: sessions, caching, auditing, administration.

Rebuilt after an accidental overwrite of the original (not recoverable); written to the interface
`api.py` and `manage.py` call:

    KnowledgeService(stack, db) with .stack .index .db
    sync_document_records(), ask(user, question, session_id), create_session(user), history(...),
    delete_session(...), list_documents(user), ingest_upload(user, filename, data, record),
    update_permissions(user, document_id, groups), delete_document(user, document_id)

Permissions always come from the authenticated `Principal` (server-side records), never from a
request field. Answers are cached per (groups, question, session state, index epoch), so any index or
permission change invalidates them.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import threading
import time
import uuid

from .access import Constraints, Principal, payload_for
from .db import Database
from .display import describe_locator
from .factory import Stack
from .loaders import DocumentReadError, load_document
from .metadata import normalize_record
from .pipeline import build_chunks, stable_id
from .rag import SessionState
from .semantic_chunking import SemanticOptions

REDACTED = "[Hidden: your access to the sources of this earlier answer has changed.]"


class NotFound(Exception):
    """Unknown resource, or one the caller may not know exists (same message either way)."""


class AccessDenied(Exception):
    pass


class _Cache:
    def __init__(self, size: int):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value) -> None:
        if self.size <= 0:
            return
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)


class KnowledgeService:
    def __init__(self, stack: Stack, db: Database):
        self.stack, self.db = stack, db
        self.index, self.settings, self.rag = stack.index, stack.settings, stack.rag
        self._cache = _Cache(self.settings.cache_size)
        self._hits: dict[str, deque] = {}
        self._rate_lock = threading.Lock()

    # ---- records ---------------------------------------------------------------------------------
    def sync_document_records(self) -> None:
        self.db.replace_documents(self.index.documents())

    # ---- helpers ---------------------------------------------------------------------------------
    def _rate_limit(self, user: Principal) -> None:
        limit = self.settings.rate_limit_per_minute
        if limit <= 0:
            return
        with self._rate_lock:
            window = self._hits.setdefault(user.user_id, deque())
            while window and window[0] < time.monotonic() - 60:
                window.popleft()
            if len(window) >= limit:
                raise ValueError("Too many requests; please wait a minute.")
            window.append(time.monotonic())

    def _require_admin(self, user: Principal) -> None:
        if not user.is_admin:
            raise AccessDenied("Administrator rights are required.")

    def _public_citation(self, citation: dict) -> dict:
        return {**citation, "display": describe_locator(citation.get("locator"), citation.get("section"))}

    # ---- sessions --------------------------------------------------------------------------------
    def create_session(self, user: Principal) -> str:
        session_id = uuid.uuid4().hex
        self.db.save_session(session_id, user.user_id, SessionState().to_dict())
        return session_id

    def _state(self, user: Principal, session_id: str) -> SessionState:
        data = self.db.load_session(session_id, user.user_id)
        if data is None:
            raise NotFound("Unknown session.")  # also returned for another user's session
        return SessionState.from_dict(data)

    def history(self, user: Principal, session_id: str) -> list[dict]:
        self._state(user, session_id)
        constraints = Constraints(user.groups)
        out = []
        for turn in self.db.turns(session_id, user.user_id):
            allowed = all((c := self.index.get_chunk(cid)) is not None and constraints.matches(payload_for(c))
                          for cid in turn["cited_chunk_ids"])
            out.append({"role": turn["role"], "content": turn["content"] if allowed else REDACTED,
                        "created_at": turn["created_at"], "redacted": not allowed})
        return out

    def delete_session(self, user: Principal, session_id: str) -> None:
        if not self.db.delete_session(session_id, user.user_id):
            raise NotFound("Unknown session.")
        self.db.audit(user.user_id, "session.delete", {"session_id": session_id})

    # ---- asking ----------------------------------------------------------------------------------
    def ask(self, user: Principal, question: str, session_id: str | None = None) -> dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if len(question) > self.settings.max_question_chars:
            raise ValueError(f"Question is longer than {self.settings.max_question_chars} characters.")
        self._rate_limit(user)
        session_id = session_id or self.create_session(user)
        state = self._state(user, session_id)
        key = (tuple(sorted(user.groups)), question.lower(), repr(state.to_dict()), self.index.epoch,
               str(self.rag.today))
        response = self._cache.get(key)
        cached = response is not None
        started = time.perf_counter()
        if response is None:
            response = self.rag.answer(question, user, state)
            self._cache.put(key, response)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        citations = [self._public_citation(c) for c in response.citations]
        self.db.save_session(session_id, user.user_id, response.state.to_dict())
        self.db.add_turn(session_id, user.user_id, "user", question)
        self.db.add_turn(session_id, user.user_id, "assistant", response.answer, [c["chunk_id"] for c in citations])
        self.db.audit(user.user_id, "ask", {
            "session_id": session_id, "question_sha256": sha256(question.encode()).hexdigest()[:16],
            "chars": len(question), "abstained": response.abstained, "citations": len(citations),
            "cached": cached, "ms": elapsed})
        return {"session_id": session_id, "answer": response.answer, "abstained": response.abstained,
                "citations": citations, "warnings": response.warnings,
                "analysis": {k: response.analysis.get(k) for k in ("family", "versions", "comparison", "followup")},
                "diagnostics": response.diagnostics, "cached": cached, "latency_ms": elapsed}

    # ---- documents -------------------------------------------------------------------------------
    def list_documents(self, user: Principal) -> list[dict]:
        return self.index.documents(Constraints(user.groups))

    def ingest_upload(self, user: Principal, filename: str, data: bytes, record: dict) -> dict:
        self._require_admin(user)
        from .pipeline import MAX_FILE_BYTES
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the upload size limit.")
        name = Path(filename).name
        source = f"uploads/{name}"
        try:
            metadata = normalize_record(source, record)
            pages, warnings = load_document(data, Path(name).suffix.lower())
        except DocumentReadError as exc:
            raise ValueError(str(exc)) from None
        if not pages:
            raise ValueError("No usable text could be extracted from the file.")
        file_hash = sha256(data).hexdigest()
        document_id = stable_id({"source": source, "file_sha256": file_hash})
        options = SemanticOptions()
        settings = {"strategy": "semantic-sentence-v1", "parser_version": 2,
                    "embedder": self.stack.embedder.fingerprint, **asdict(options)}
        chunks = build_chunks(pages, document_id=document_id, source=source, file_hash=file_hash,
                              metadata_hash=stable_id(metadata), metadata=metadata, strategy="semantic",
                              settings=settings, max_words=200, overlap_words=40,
                              embedder=self.stack.embedder, options=options)
        for old in self.index.documents():
            if old["source"] == source and old["document_id"] != document_id:
                self.index.delete_document(old["document_id"])  # same file name = replacement
        self.index.add_document(chunks)
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.upload_dir / f"{document_id[:12]}_{name}").write_bytes(data)
        self.sync_document_records()
        self.db.audit(user.user_id, "doc.upload", {"document_id": document_id, "source": source, "chunks": len(chunks)})
        return {"document_id": document_id, "source": source, "chunks": len(chunks), "warnings": warnings}

    def update_permissions(self, user: Principal, document_id: str, groups: list[str]) -> dict:
        self._require_admin(user)
        try:
            updated = self.index.update_document_metadata(document_id, {"access_groups": groups})
        except KeyError:
            raise NotFound("Unknown document.") from None
        self.sync_document_records()
        self.db.audit(user.user_id, "doc.permissions", {"document_id": document_id, "access_groups": updated["access_groups"]})
        return {"document_id": document_id, "access_groups": updated["access_groups"]}

    def delete_document(self, user: Principal, document_id: str) -> None:
        self._require_admin(user)
        if not self.index.delete_document(document_id):
            raise NotFound("Unknown document.")
        self.sync_document_records()
        self.db.audit(user.user_id, "doc.delete", {"document_id": document_id})
