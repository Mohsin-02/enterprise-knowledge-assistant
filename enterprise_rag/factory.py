"""Assemble the full stack (embedder, vector store, BM25, reranker, LLM, RAG pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .index_manager import IndexManager
from .llm import OpenAICompatibleLLM
from .pipeline import ingest
from .rag import RAGPipeline
from .rerank import make_reranker
from .retrieval import HybridRetriever
from .settings import Settings
from .vector_store import InMemoryVectorStore, QdrantVectorStore


@dataclass
class Stack:
    settings: Settings
    embedder: object
    index: IndexManager
    retriever: HybridRetriever
    rag: RAGPipeline


def make_embedder(kind: str):
    kind = kind.lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "bge-small":
        return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    if kind == "bge-m3":
        return SentenceTransformerEmbedder("BAAI/bge-m3", max_seq_length=512, batch_size=8)
    if kind.startswith("st:"):
        return SentenceTransformerEmbedder(kind[3:])
    raise ValueError(f"Unknown embedder {kind!r}: use hashing, bge-small, bge-m3 or st:<model>.")


def make_vector_store(settings: Settings, dim: int):
    if settings.vector_store == "memory":
        return InMemoryVectorStore(dim)
    if settings.vector_store == "qdrant":
        return QdrantVectorStore(dim, url=settings.qdrant_url, path=settings.qdrant_path,
                                 collection=settings.qdrant_collection)
    raise ValueError("vector_store must be 'memory' or 'qdrant'.")


def make_llm(settings: Settings):
    if not settings.llm_url:
        return None  # offline extractive answers
    return OpenAICompatibleLLM(settings.llm_url, settings.llm_model, settings.llm_api_key)


def _read_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_stack(settings: Settings | None = None, *, today: date | None = None,
                rebuild: bool = False, embedder=None) -> Stack:
    """Load the index. Source of truth, in order: `index_dir/chunks.jsonl` (includes uploads and
    metadata edits), else a fresh ingestion of `data_raw` into `processed_dir`."""
    settings = settings or Settings.from_env()
    embedder = embedder or make_embedder(settings.embedder)
    repository = settings.index_dir / "chunks.jsonl"
    if rebuild or not repository.exists():
        strategy = settings.chunk_strategy
        ingest(settings.data_raw, settings.processed_dir, metadata_path=settings.registry, strategy=strategy,
               embedder=embedder if strategy == "semantic" else None)
        chunks = _read_chunks(settings.processed_dir / "chunks.jsonl")
        from_fresh = True
    else:
        chunks = _read_chunks(repository)
        from_fresh = False
    index = IndexManager(embedder, make_vector_store(settings, embedder.dim), index_dir=settings.index_dir)
    index.load_chunks(chunks)
    if from_fresh:
        index.save()
    retriever = HybridRetriever(index, make_reranker(settings.reranker, settings.reranker_model))
    rag = RAGPipeline(index, retriever, make_llm(settings), today=today)
    return Stack(settings, embedder, index, retriever, rag)
