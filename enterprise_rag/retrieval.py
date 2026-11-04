"""Hybrid retrieval: dense + BM25 under identical constraints, RRF fusion, optional reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from .access import Constraints
from .fusion import reciprocal_rank_fusion
from .index_manager import IndexManager, embedding_text
from .rerank import Reranker


@dataclass(frozen=True)
class RetrievalOptions:
    mode: str = "hybrid"        # "hybrid" | "dense" | "bm25"
    rerank: bool = True
    dense_k: int = 30
    bm25_k: int = 30
    fused_k: int = 20
    final_k: int = 5
    rrf_k: int = 60


@dataclass
class Evidence:
    chunk: dict
    score: float
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    def public(self) -> dict:
        c = self.chunk
        return {"chunk_id": c["chunk_id"], "source": c["source"], "title": c["title"],
                "document_family": c["document_family"], "version": c["version"],
                "effective_from": c.get("effective_from"), "effective_to": c.get("effective_to"),
                "locator": c.get("locator"), "section": c.get("section_path"), "text": c["text"],
                "score": round(self.score, 4)}


@dataclass
class RetrievalResult:
    evidence: list[Evidence]
    diagnostics: dict = field(default_factory=dict)
    dense_top: float = 0.0


class HybridRetriever:
    def __init__(self, index: IndexManager, reranker: Reranker | None = None):
        self.index, self.reranker = index, reranker

    def retrieve(self, query: str, constraints: Constraints, options: RetrievalOptions | None = None
                 ) -> RetrievalResult:
        opts = options or RetrievalOptions()
        if opts.mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError("mode must be hybrid, dense or bm25.")
        diag: dict = {"mode": opts.mode, "rerank": bool(opts.rerank and self.reranker)}
        if not constraints.groups or not query.strip():
            return RetrievalResult([], {**diag, "note": "no searchable scope"})
        t0 = time.perf_counter()
        dense_hits: list[tuple[str, float]] = []
        bm25_hits: list[tuple[str, float]] = []
        if opts.mode in {"hybrid", "dense"}:
            dense_hits = self.index.vectors.search(self.index.embedder.embed_query(query), constraints, opts.dense_k)
        t1 = time.perf_counter()
        if opts.mode in {"hybrid", "bm25"}:
            bm25_hits = self.index.lexical.search(query, constraints, opts.bm25_k)
        t2 = time.perf_counter()
        dense_scores = dict(dense_hits)
        dense_rank = {cid: i for i, (cid, _) in enumerate(dense_hits, 1)}
        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_hits, 1)}
        if opts.mode == "hybrid":
            fused = reciprocal_rank_fusion({"dense": [c for c, _ in dense_hits], "bm25": [c for c, _ in bm25_hits]},
                                           k=opts.rrf_k)
            ordered = [(f.chunk_id, f.score) for f in fused[:opts.fused_k]]
        elif opts.mode == "dense":
            ordered = dense_hits[:opts.fused_k]
        else:
            ordered = bm25_hits[:opts.fused_k]
        candidates: list[Evidence] = []
        for chunk_id, score in ordered:
            chunk = self.index.get_chunk(chunk_id)
            if chunk is None or not self.index.allowed(chunk, constraints):  # defence in depth
                continue
            candidates.append(Evidence(chunk, score, dense_scores.get(chunk_id), dense_rank.get(chunk_id),
                                       bm25_rank.get(chunk_id), score if opts.mode == "hybrid" else None))
        t3 = time.perf_counter()
        if opts.rerank and self.reranker and candidates:
            scores = self.reranker.score(query, [embedding_text(e.chunk) for e in candidates])
            for evidence, score in zip(candidates, scores):
                evidence.rerank_score = score
                evidence.score = score
            candidates.sort(key=lambda e: -e.rerank_score)  # stable: ties keep fused order
        t4 = time.perf_counter()
        diag.update({
            "dense_candidates": len(dense_hits), "bm25_candidates": len(bm25_hits),
            "fused_candidates": len(candidates),
            "latency_ms": {"dense": round((t1 - t0) * 1000, 2), "bm25": round((t2 - t1) * 1000, 2),
                           "fuse": round((t3 - t2) * 1000, 2), "rerank": round((t4 - t3) * 1000, 2),
                           "total": round((t4 - t0) * 1000, 2)}})
        return RetrievalResult(candidates[:opts.final_k], diag, max(dense_scores.values(), default=0.0))
