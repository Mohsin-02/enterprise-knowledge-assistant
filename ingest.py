"""Run from the project root: python ingest.py"""

import argparse
from pathlib import Path

from enterprise_rag.pipeline import ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="Enterprise RAG: parse documents and build chunks.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata", type=Path, default=Path("config/documents.json"))
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--strategy", choices=["word-window", "semantic"], default="word-window",
                        help="word-window is the Step 1 baseline; semantic is structure/similarity aware")
    args = parser.parse_args()
    try:
        manifest = ingest(args.input, args.output, args.chunk_words, args.overlap_words,
                          metadata_path=args.metadata, strategy=args.strategy)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ingestion stopped: {exc}\n")

    summary = manifest["summary"]
    print(f"INGESTION COMPLETE (strategy: {args.strategy})")
    print(f"Documents found:   {summary['input_documents']}")
    print(f"Documents indexed: {summary['indexed_documents']}")
    print(f"Duplicate files:   {summary['duplicate_documents']}")
    print(f"Empty documents:   {summary['empty_documents']}")
    print(f"Failed documents:  {summary['failed_documents']}")
    print(f"Unregistered:      {summary['unregistered_documents']}")
    print(f"Chunks created:    {summary['total_chunks']}")
    print(f"Chunks:   {args.output / 'chunks.jsonl'}")
    print(f"Manifest: {args.output / 'manifest.json'}")
    for document in manifest["documents"]:
        for warning in document["warnings"]:
            print(f"WARNING [{document['source']}]: {warning}")
        if "error" in document:
            print(f"ERROR [{document['source']}]: {document['error']}")
    if summary["failed_documents"] or summary["empty_documents"] or summary["unregistered_documents"]:
        print("Some documents were not indexed. Check manifest.json before proceeding.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
