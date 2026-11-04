"""Runtime configuration, read from ERAG_* environment variables.

Rebuilt after an accidental overwrite of the original (not recoverable). Field names follow what
`factory.py`, `manage.py`, `api.py` and `evaluate.py` use.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import os
from pathlib import Path


@dataclass
class Settings:
    database_url: str = "sqlite:///data/app.db"      # or postgresql://user:pass@host:5432/db
    data_raw: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    registry: Path = Path("config/documents.json")
    index_dir: Path = Path("data/index")
    upload_dir: Path = Path("data/uploads")
    chunk_strategy: str = "semantic"                 # semantic | word-window
    embedder: str = "hashing"                        # hashing | bge-small | bge-m3 | st:<model>
    vector_store: str = "memory"                     # memory | qdrant
    qdrant_url: str | None = None
    qdrant_path: str | None = None
    qdrant_collection: str = "enterprise_chunks"
    reranker: str = "lexical"                        # none | lexical | cross-encoder
    reranker_model: str | None = None
    llm_url: str | None = None                       # e.g. http://localhost:11434/v1 (Ollama); empty = extractive answers
    llm_model: str = "qwen2.5:3b-instruct"
    llm_api_key: str | None = None
    rate_limit_per_minute: int = 60                  # per user; 0 disables
    cache_size: int = 256
    max_question_chars: int = 2000

    @classmethod
    def from_env(cls) -> "Settings":
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f"ERAG_{f.name.upper()}")
            if raw is None or (raw == "" and f.type.startswith("str | None")):
                continue
            if f.type.startswith("int"):
                values[f.name] = int(raw)
            elif f.type.startswith("Path"):
                values[f.name] = Path(raw)
            else:
                values[f.name] = raw or None
        return cls(**values)
