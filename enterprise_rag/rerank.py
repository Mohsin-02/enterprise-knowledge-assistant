"""Rerankers score (query, passage) pairs. The cross-encoder is the real one; the lexical
heuristic is an offline stand-in and is not a substitute for it in quality comparisons."""

from __future__ import annotations

from typing import Protocol, Sequence

from .textnorm import expand_terms, numbers_in, tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LexicalReranker:
    name = "lexical-overlap-heuristic"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q_terms = tokenize(query)
        wanted = set(expand_terms(q_terms))
        q_bigrams = set(zip(q_terms, q_terms[1:]))
        q_numbers = numbers_in(query)
        scores = []
        for passage in passages:
            p_terms = tokenize(passage)
            present = set(p_terms)
            coverage = len({t for t in q_terms if t in present or any(
                s in present for s in expand_terms([t])[1:])}) / (len(set(q_terms)) or 1)
            bigram_hits = len(q_bigrams & set(zip(p_terms, p_terms[1:]))) / (len(q_bigrams) or 1)
            number_hits = len(q_numbers & numbers_in(passage)) / (len(q_numbers) or 1) if q_numbers else 0.0
            scores.append(coverage + 0.5 * bigram_hits + 0.2 * number_hits + 0.05 * len(wanted & present) / (len(wanted) or 1))
        return scores


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder. Default is small enough for a CPU-only 8 GB laptop;
    BAAI/bge-reranker-base is a stronger, slower alternative."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
        from sentence_transformers import CrossEncoder  # optional dependency

        self._model = CrossEncoder(model_name, max_length=max_length)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages], show_progress_bar=False)]


def make_reranker(kind: str, model_name: str | None = None) -> Reranker | None:
    kind = kind.lower()
    if kind in {"none", "off"}:
        return None
    if kind == "lexical":
        return LexicalReranker()
    if kind in {"cross-encoder", "crossencoder"}:
        return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    raise ValueError(f"Unknown reranker {kind!r}; use none, lexical or cross-encoder.")
