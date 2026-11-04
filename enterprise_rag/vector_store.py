"""Vector stores: an exact in-memory store (tests, small corpora) and Qdrant (the target store).

Both apply the same `Constraints` BEFORE ranking. The Qdrant implementation could not be executed in
the assistant's sandbox (no network); `tests/test_optional_backends.py` exercises it on your machine
using Qdrant's in-process mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import uuid

import numpy as np

from .access import Constraints


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, items: list[tuple[str, np.ndarray, dict]]) -> None: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    @abstractmethod
    def update_document_payload(self, document_id: str, fields: dict) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, constraints: Constraints, k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    def chunk_ids(self) -> set[str]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._payload: dict[str, dict] = {}

    def count(self) -> int:
        return len(self._ids)

    def chunk_ids(self) -> set[str]:
        return set(self._ids)

    def upsert(self, items):
        if not items:
            return
        incoming = {cid for cid, _, _ in items}
        keep = [i for i, cid in enumerate(self._ids) if cid not in incoming]
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.stack([np.asarray(v, dtype=np.float32) for _, v, _ in items])
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match store dimension {self.dim}.")
        self._matrix = np.vstack([self._matrix, vectors])
        for cid, _, payload in items:
            self._ids.append(cid)
            self._payload[cid] = dict(payload)

    def delete_document(self, document_id: str) -> None:
        keep = [i for i, cid in enumerate(self._ids) if self._payload[cid]["document_id"] != document_id]
        for i, cid in enumerate(self._ids):
            if i not in set(keep):
                self._payload.pop(cid, None)
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        drop = set(chunk_ids)
        keep = [i for i, cid in enumerate(self._ids) if cid not in drop]
        for cid in drop:
            self._payload.pop(cid, None)
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def update_document_payload(self, document_id: str, fields: dict) -> None:
        for payload in self._payload.values():
            if payload["document_id"] == document_id:
                payload.update(fields)

    def search(self, vector, constraints, k):
        mask = [i for i, cid in enumerate(self._ids) if constraints.matches(self._payload[cid])]
        if not mask:
            return []
        scores = self._matrix[mask] @ np.asarray(vector, dtype=np.float32)
        order = sorted(range(len(mask)), key=lambda j: (-float(scores[j]), self._ids[mask[j]]))[:k]
        return [(self._ids[mask[j]], float(scores[j])) for j in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", ids=np.array(self._ids), matrix=self._matrix)

    def load(self, directory: Path, payloads: dict[str, dict]) -> bool:
        path = directory / "vectors.npz"
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=False)
        ids = [str(i) for i in data["ids"]]
        if set(ids) != set(payloads) or data["matrix"].shape[1] != self.dim:
            return False
        self._ids, self._matrix = ids, data["matrix"].astype(np.float32)
        self._payload = {cid: dict(payloads[cid]) for cid in ids}
        return True


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store. Use url="http://localhost:6333", path="dir" (embedded), or ":memory:"."""

    def __init__(self, dim: int, *, url: str | None = None, path: str | None = None,
                 collection: str = "enterprise_chunks", api_key: str | None = None):
        from qdrant_client import QdrantClient, models

        self._m = models
        self.collection, self.dim = collection, dim
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=path) if path and path != ":memory:" else QdrantClient(":memory:")
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        else:
            info = self.client.get_collection(collection)
            size = info.config.params.vectors.size
            if size != dim:
                raise ValueError(f"Collection {collection!r} has dimension {size}; embedder produces {dim}. "
                                 "Use a new collection when changing embedding models.")
        for field, schema in (("access_groups", "keyword"), ("document_family", "keyword"),
                              ("version", "keyword"), ("document_id", "keyword"),
                              ("effective_from_ord", "integer"), ("effective_to_ord", "integer")):
            try:
                self.client.create_payload_index(collection, field_name=field, field_schema=schema)
            except Exception:
                pass

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.UUID(hex=chunk_id[:32]))

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def chunk_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset,
                                                with_payload=["chunk_id"], with_vectors=False)
            ids.update(p.payload["chunk_id"] for p in points)
            if offset is None:
                return ids

    def _filter(self, constraints: Constraints):
        m = self._m
        must = [m.FieldCondition(key="metadata_registered", match=m.MatchValue(value=True)),
                m.FieldCondition(key="access_groups", match=m.MatchAny(any=sorted(constraints.groups)))]
        for key, values in (("document_family", constraints.families), ("version", constraints.versions),
                            ("document_id", constraints.document_ids)):
            if values is not None:
                must.append(m.FieldCondition(key=key, match=m.MatchAny(any=sorted(values))))
        if constraints.as_of is not None:
            day = constraints.as_of.toordinal()
            must.append(m.FieldCondition(key="effective_from_ord", range=m.Range(lte=day)))
            must.append(m.FieldCondition(key="effective_to_ord", range=m.Range(gte=day)))
        return m.Filter(must=must)

    def _doc_filter(self, document_id: str):
        m = self._m
        return m.Filter(must=[m.FieldCondition(key="document_id", match=m.MatchValue(value=document_id))])

    def upsert(self, items):
        m = self._m
        for i in range(0, len(items), 128):
            batch = items[i:i + 128]
            self.client.upsert(self.collection, points=[
                m.PointStruct(id=self.point_id(cid), vector=np.asarray(vec, dtype=np.float32).tolist(),
                              payload=payload) for cid, vec, payload in batch])

    def delete_document(self, document_id: str) -> None:
        self.client.delete(self.collection, points_selector=self._m.FilterSelector(
            filter=self._doc_filter(document_id)))

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.client.delete(self.collection, points_selector=self._m.PointIdsList(
                points=[self.point_id(c) for c in chunk_ids]))

    def update_document_payload(self, document_id: str, fields: dict) -> None:
        self.client.set_payload(self.collection, payload=fields, points=self._doc_filter(document_id))

    def search(self, vector, constraints, k):
        if not constraints.groups:
            return []
        response = self.client.query_points(
            self.collection, query=np.asarray(vector, dtype=np.float32).tolist(),
            query_filter=self._filter(constraints), limit=k, with_payload=["chunk_id"])
        return [(p.payload["chunk_id"], float(p.score)) for p in response.points]
