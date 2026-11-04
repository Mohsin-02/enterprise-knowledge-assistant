"""BM25 keyword retrieval with permission-aware statistics.

Corpus statistics (document frequency, average length, N) are computed over the chunks the caller is
allowed to see, so scores never encode information about restricted documents. Allowed-set
computations are cached per constraint and cleared on every index mutation.
"""

from __future__ import annotations

from collections import Counter
import math

from .access import Constraints
from .textnorm import expand_terms, tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._payload: dict[str, dict] = {}
        self._length: dict[str, int] = {}
        self._terms: dict[str, Counter] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._by_document: dict[str, set[str]] = {}
        self._allowed: dict[tuple, tuple[frozenset[str], float]] = {}

    def __len__(self) -> int:
        return len(self._payload)

    def add(self, chunk_id: str, text: str, payload: dict) -> None:
        if chunk_id in self._payload:
            self._remove_chunk(chunk_id)
        counts = Counter(tokenize(text))
        self._payload[chunk_id] = payload
        self._terms[chunk_id] = counts
        self._length[chunk_id] = sum(counts.values())
        for term, tf in counts.items():
            self._postings.setdefault(term, {})[chunk_id] = tf
        self._by_document.setdefault(payload["document_id"], set()).add(chunk_id)
        self._allowed.clear()

    def _remove_chunk(self, chunk_id: str) -> None:
        payload = self._payload.pop(chunk_id)
        for term in self._terms.pop(chunk_id):
            posting = self._postings[term]
            posting.pop(chunk_id, None)
            if not posting:
                del self._postings[term]
        self._length.pop(chunk_id, None)
        self._by_document[payload["document_id"]].discard(chunk_id)

    def remove_document(self, document_id: str) -> None:
        for chunk_id in list(self._by_document.get(document_id, ())):
            self._remove_chunk(chunk_id)
        self._by_document.pop(document_id, None)
        self._allowed.clear()

    def update_document_payload(self, document_id: str, fields: dict) -> None:
        for chunk_id in self._by_document.get(document_id, ()):
            self._payload[chunk_id].update(fields)
        self._allowed.clear()

    def _allowed_set(self, constraints: Constraints) -> tuple[frozenset[str], float]:
        key = constraints.cache_key()
        if key not in self._allowed:
            ids = frozenset(cid for cid, payload in self._payload.items() if constraints.matches(payload))
            average = (sum(self._length[c] for c in ids) / len(ids)) if ids else 0.0
            self._allowed[key] = (ids, average)
        return self._allowed[key]

    def term_weights(self, terms: list[str], constraints: Constraints) -> dict[str, float]:
        """IDF over the allowed subset; terms absent from it get the maximum weight."""
        allowed, _ = self._allowed_set(constraints)
        n = len(allowed)
        weights: dict[str, float] = {}
        for term in set(terms):
            df = sum(1 for cid in self._postings.get(term, ()) if cid in allowed)
            weights[term] = math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0
        return weights

    def search(self, query: str, constraints: Constraints, k: int = 20) -> list[tuple[str, float]]:
        allowed, average = self._allowed_set(constraints)
        if not allowed:
            return []
        terms = list(dict.fromkeys(expand_terms(tokenize(query))))
        weights = self.term_weights(terms, constraints)
        scores: dict[str, float] = {}
        for term in terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = weights[term]
            for chunk_id, tf in posting.items():
                if chunk_id not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._length[chunk_id] / (average or 1.0))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
