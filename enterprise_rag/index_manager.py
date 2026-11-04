"""Owns the chunk repository and keeps the vector store and BM25 index consistent with it.

The chunk repository (chunks.jsonl) is the source of truth for text and metadata. Vector and lexical
payloads are derived from it, so a metadata-only change (for example access groups) updates payloads
without re-embedding, and retrieval re-checks every candidate against the repository.
`epoch` increases on every mutation; caches key on it so stale answers are never served.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

from .access import Constraints, payload_for
from .embeddings import Embedder
from .lexical import BM25Index
from .metadata import normalize_record
from .pipeline import atomic_write
from .vector_store import InMemoryVectorStore, VectorStore

METADATA_FIELDS = ("title", "document_family", "version", "effective_from", "effective_to", "access_groups")


def embedding_text(chunk: dict) -> str:
    return chunk.get("contextual_text") or f"{chunk.get('title', '')}\n{chunk['text']}"


class IndexManager:
    def __init__(self, embedder: Embedder, vectors: VectorStore, lexical: BM25Index | None = None,
                 index_dir: Path | None = None, batch_size: int = 32):
        self.embedder, self.vectors = embedder, vectors
        self.lexical = lexical or BM25Index()
        self.index_dir = Path(index_dir) if index_dir else None
        self.batch_size = batch_size
        self.chunks: dict[str, dict] = {}
        self._by_document: dict[str, list[str]] = {}
        self.epoch = 0
        self.lock = threading.RLock()

    # ---- loading -------------------------------------------------------------------------------
    def load_chunks(self, chunks: list[dict]) -> None:
        """Replace the repository, reusing stored vectors when their fingerprint and IDs match."""
        with self.lock:
            self.chunks = {c["chunk_id"]: c for c in chunks}
            self._by_document = {}
            for chunk in chunks:
                self._by_document.setdefault(chunk["document_id"], []).append(chunk["chunk_id"])
            payloads = {cid: payload_for(c) for cid, c in self.chunks.items()}
            for cid, chunk in self.chunks.items():
                self.lexical.add(cid, embedding_text(chunk), payloads[cid])
            reused = False
            if self.index_dir and isinstance(self.vectors, InMemoryVectorStore):
                meta = self._read_meta()
                if meta.get("embedder") == self.embedder.fingerprint:
                    reused = self.vectors.load(self.index_dir, payloads)
            if not reused:
                stale = self._stale_vector_ids()
                if stale:
                    self.vectors.delete_chunks(sorted(stale))
                present = self._present_vector_ids()
                self._embed_and_store([c for cid, c in self.chunks.items() if cid not in present])
            self.epoch += 1

    def _present_vector_ids(self) -> set[str]:
        try:
            return self.vectors.chunk_ids() & set(self.chunks)
        except NotImplementedError:
            return set()

    def _stale_vector_ids(self) -> set[str]:
        try:
            return self.vectors.chunk_ids() - set(self.chunks)
        except NotImplementedError:
            return set()

    def _embed_and_store(self, chunks: list[dict]) -> None:
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            vectors = self.embedder.embed_documents([embedding_text(c) for c in batch])
            self.vectors.upsert([(c["chunk_id"], v, payload_for(c)) for c, v in zip(batch, vectors)])

    def _read_meta(self) -> dict:
        try:
            return json.loads((self.index_dir / "index_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    def save(self) -> None:
        if not self.index_dir:
            return
        with self.lock:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            lines = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in self.chunks.values())
            atomic_write(self.index_dir / "chunks.jsonl", lines.encode("utf-8"))
            atomic_write(self.index_dir / "index_meta.json", (json.dumps(
                {"embedder": self.embedder.fingerprint, "chunks": len(self.chunks)}, indent=2) + "\n").encode())
            if isinstance(self.vectors, InMemoryVectorStore):
                self.vectors.save(self.index_dir)

    # ---- mutation ------------------------------------------------------------------------------
    def add_document(self, chunks: list[dict]) -> None:
        """Add or replace one document's chunks. Other versions of the family are retained."""
        if not chunks:
            raise ValueError("A document must have at least one chunk.")
        document_id = chunks[0]["document_id"]
        with self.lock:
            self._drop_document(document_id)
            for chunk in chunks:
                self.chunks[chunk["chunk_id"]] = chunk
                self.lexical.add(chunk["chunk_id"], embedding_text(chunk), payload_for(chunk))
            self._by_document[document_id] = [c["chunk_id"] for c in chunks]
            self._embed_and_store(chunks)
            self.epoch += 1
            self.save()

    def _drop_document(self, document_id: str) -> None:
        for cid in self._by_document.pop(document_id, []):
            self.chunks.pop(cid, None)
        self.lexical.remove_document(document_id)
        self.vectors.delete_document(document_id)

    def delete_document(self, document_id: str) -> bool:
        with self.lock:
            existed = document_id in self._by_document
            self._drop_document(document_id)
            self.epoch += 1
            self.save()
            return existed

    def update_document_metadata(self, document_id: str, changes: dict) -> dict:
        """Metadata-only update: refresh stored chunks and payloads without re-embedding."""
        unknown = set(changes) - set(METADATA_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported metadata fields: {sorted(unknown)}")
        with self.lock:
            ids = self._by_document.get(document_id)
            if not ids:
                raise KeyError(document_id)
            first = self.chunks[ids[0]]
            current = {f: first.get(f) for f in METADATA_FIELDS}
            updated = normalize_record(first["source"], {**current, **changes})
            for cid in ids:
                self.chunks[cid].update(updated)
            fields = payload_for(self.chunks[ids[0]])
            for key in ("chunk_id",):
                fields.pop(key)
            self.vectors.update_document_payload(document_id, fields)
            self.lexical.update_document_payload(document_id, fields)
            self.epoch += 1
            self.save()
            return updated

    # ---- reading -------------------------------------------------------------------------------
    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.chunks.get(chunk_id)

    def allowed(self, chunk: dict, constraints: Constraints) -> bool:
        return constraints.matches(payload_for(chunk))

    def documents(self, constraints: Constraints | None = None) -> list[dict]:
        out = []
        for document_id, ids in self._by_document.items():
            first = self.chunks[ids[0]]
            if constraints is not None and not constraints.matches(payload_for(first)):
                continue
            out.append({"document_id": document_id, "source": first["source"],
                        **{f: first.get(f) for f in METADATA_FIELDS}, "chunks": len(ids)})
        return sorted(out, key=lambda d: (d["document_family"], d["effective_from"] or "", d["source"]))

    def chunks_for(self, constraints: Constraints, limit: int | None = None) -> list[dict]:
        found = [c for c in self.chunks.values() if constraints.matches(payload_for(c))]
        found.sort(key=lambda c: (c["source"], c["chunk_index"]))
        return found[:limit] if limit else found
