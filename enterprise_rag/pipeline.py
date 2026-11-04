"""Build a deterministic corpus of chunks and an inspectable ingestion manifest."""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from .chunking import split_text
from .loaders import DocumentReadError, SUPPORTED_SUFFIXES, load_document
from .metadata import load_registry, metadata_for_source
from .semantic_chunking import SemanticOptions, semantic_chunk_blocks


MAX_FILE_BYTES = 20 * 1024 * 1024


def stable_id(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    """Replace one output file only after its complete content has been written."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(data)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_chunks(pages, *, document_id: str, source: str, file_hash: str, metadata_hash: str,
                 metadata: dict, strategy: str, settings: dict, max_words: int, overlap_words: int,
                 embedder=None, options: SemanticOptions | None = None) -> list[dict]:
    """Turn parsed blocks into chunk records with stable IDs (shared by batch and upload paths)."""
    base = {"document_id": document_id, "source": source, "file_sha256": file_hash,
            "metadata_sha256": metadata_hash}
    chunks: list[dict] = []
    if strategy == "semantic":
        for ordinal, sc in enumerate(semantic_chunk_blocks(pages, title=metadata["title"], embedder=embedder,
                                                           options=options)):
            words = len(sc.text.split())
            chunk_id = stable_id({
                "document_id": document_id, "block_index": sc.first_block, "locator": sc.locator,
                "char_start": sc.char_start, "char_end": sc.char_end, "ordinal": ordinal, "settings": settings})
            chunks.append({
                "chunk_id": chunk_id, **base, **metadata, "page": sc.page, "block_index": sc.first_block,
                "locator": sc.locator, "chunk_index": ordinal, "word_count": words, "text": sc.text,
                "contextual_text": sc.contextual_text, "section_path": sc.section_path,
                "token_count": sc.token_count, "word_start": 0, "word_end": words,
                "char_start": sc.char_start, "char_end": sc.char_end})
        return chunks
    chunk_index = 0
    for block_index, page in enumerate(pages):
        for chunk in split_text(page.text, max_words, overlap_words):
            chunk_id = stable_id({
                "document_id": document_id, "page": page.number, "block_index": block_index,
                "locator": page.locator, "char_start": chunk.char_start, "char_end": chunk.char_end,
                "settings": settings,
            })
            chunks.append({
                "chunk_id": chunk_id, **base, **metadata, "page": page.number, "block_index": block_index,
                "locator": page.locator, "chunk_index": chunk_index, "word_count": chunk.word_count,
                **asdict(chunk),
            })
            chunk_index += 1
    return chunks


def ingest(
    input_dir: Path,
    output_dir: Path,
    max_words: int = 200,
    overlap_words: int = 40,
    *,
    metadata_path: Path | None = None,
    strategy: str = "word-window",
    embedder=None,
    semantic_options: dict | None = None,
) -> dict:
    """Rebuild the current corpus. `strategy` is "word-window" (baseline) or "semantic"."""
    if strategy not in {"word-window", "semantic"}:
        raise ValueError("strategy must be 'word-window' or 'semantic'.")
    split_text("", max_words, overlap_words)  # Validate settings before writing.
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Input and output must be separate, non-nested directories.")
    registry = load_registry(metadata_path)

    paths = sorted(
        (path for path in input_dir.rglob("*")
         if path.is_file() and not path.is_symlink()
         and input_dir in path.resolve().parents
         and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )
    if not paths:
        raise ValueError(f"No supported files found ({', '.join(sorted(SUPPORTED_SUFFIXES))}). Existing outputs were kept.")

    documents: list[dict] = []
    chunks: list[dict] = []
    seen_hashes: dict[tuple[str, str], str] = {}
    options = None
    if strategy == "semantic":
        from .embeddings import HashingEmbedder
        embedder = embedder or HashingEmbedder()
        options = SemanticOptions(**(semantic_options or {}))
        settings = {"strategy": "semantic-sentence-v1", "parser_version": 2,
                    "embedder": embedder.fingerprint, **asdict(options)}
    else:
        settings = {"strategy": "word-window-per-block-v2", "parser_version": 2, "max_words": max_words,
                    "overlap_words": overlap_words}

    for path in paths:
        source = path.relative_to(input_dir).as_posix()
        document: dict = {"source": source, "status": "error", "chunks": 0, "warnings": []}
        try:
            metadata = metadata_for_source(source, registry)
            metadata_hash = stable_id(metadata)
            document.update(metadata)
            document["metadata_sha256"] = metadata_hash
            if not metadata["metadata_registered"]:
                document["warnings"].append(
                    "Unregistered metadata: this document must not be served by later retrieval."
                )
            if path.stat().st_size > MAX_FILE_BYTES:
                raise DocumentReadError("File exceeds the 20 MiB limit for this starter.")
            data = path.read_bytes()
            file_hash = sha256(data).hexdigest()
            document.update(file_bytes=len(data), file_sha256=file_hash)
            dedup_key = (file_hash, metadata_hash)
            if dedup_key in seen_hashes:
                document.update(status="duplicate", duplicate_of=seen_hashes[dedup_key])
                documents.append(document)
                continue

            pages, warnings = load_document(data, path.suffix.lower())
            document["warnings"].extend(warnings)
            if not pages:
                document["status"] = "empty"
                document["warnings"].append("No usable text was extracted from this document.")
                documents.append(document)
                continue

            document_id = stable_id({"source": source, "file_sha256": file_hash})
            document["document_id"] = document_id
            built = build_chunks(
                pages, document_id=document_id, source=source, file_hash=file_hash,
                metadata_hash=metadata_hash, metadata=metadata, strategy=strategy, settings=settings,
                max_words=max_words, overlap_words=overlap_words, embedder=embedder, options=options)
            chunks.extend(built)
            chunk_index = len(built)
            document.update(status="indexed", chunks=chunk_index)
            seen_hashes[dedup_key] = source
        except (OSError, DocumentReadError) as exc:
            document["error"] = str(exc)
        documents.append(document)

    if not chunks:
        details = "; ".join(f"{d['source']}: {d.get('error', d['status'])}" for d in documents)
        raise ValueError(f"No usable chunks produced. Existing outputs were kept. {details}")

    chunk_bytes = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks).encode("utf-8")
    manifest = {
        "schema_version": 3,
        "chunk_settings": settings,
        "chunks_sha256": sha256(chunk_bytes).hexdigest(),
        "summary": {
            "input_documents": len(documents),
            "indexed_documents": sum(d["status"] == "indexed" for d in documents),
            "duplicate_documents": sum(d["status"] == "duplicate" for d in documents),
            "empty_documents": sum(d["status"] == "empty" for d in documents),
            "failed_documents": sum(d["status"] == "error" for d in documents),
            "unregistered_documents": sum(
                d["status"] == "indexed" and not d.get("metadata_registered", False)
                for d in documents
            ),
            "total_chunks": len(chunks),
        },
        "documents": documents,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "chunks.jsonl", chunk_bytes)
    atomic_write(output_dir / "manifest.json",
                 (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest
