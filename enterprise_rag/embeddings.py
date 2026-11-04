"""Embedding backends. Documents and queries must always use the same backend/model id."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic offline stand-in: hashed word and character n-grams, L2-normalized.

    It needs no download and makes the whole pipeline runnable anywhere, but it is a lexical
    representation. It does NOT understand paraphrases the way BGE does; never report its
    metrics as if they were a neural embedding model's.
    """

    def __init__(self, dim: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.model_id = f"hashing-ngrams-v1-{dim}"
        common = dict(n_features=dim, alternate_sign=False, norm=None, lowercase=True)
        self._word = HashingVectorizer(**{"analyzer": "word", "ngram_range": (1, 2), "stop_words": "english",
                                          "token_pattern": r"(?u)\b\w+\b"}, **common)
        self._char = HashingVectorizer(**{"analyzer": "char_wb", "ngram_range": (3, 5)}, **common)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        word = self._word.transform(texts).toarray()
        char = self._char.transform(texts).toarray()
        return _l2(np.log1p(word) + 0.5 * np.log1p(char))

    @property
    def fingerprint(self) -> str:
        """Alias of ``model_id``: `index_manager` and `pipeline` read this name."""
        return self.model_id

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]


class SentenceTransformerEmbedder:
    """Real neural embeddings, e.g. BAAI/bge-small-en-v1.5 (CPU baseline) or BAAI/bge-m3 (dense)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str | None = None,
                 max_seq_length: int = 512, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.model_id = f"st:{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        if query_prefix is None and "bge" in model_name.lower() and "m3" not in model_name.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        self.query_prefix = query_prefix or ""

    @property
    def fingerprint(self) -> str:
        return self.model_id

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True,
                                 show_progress_bar=False)[0].astype(np.float32)
