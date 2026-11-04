"""Show a few saved passages and their exact source-location metadata."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect parsed evidence chunks.")
    parser.add_argument("--format", choices=["pdf", "txt", "md", "docx", "html", "xlsx"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--file", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    try:
        shown = 0
        with args.file.open(encoding="utf-8") as handle:
            for line in handle:
                chunk = json.loads(line)
                locator = chunk.get("locator") or {}
                if args.format and locator.get("format") != args.format:
                    continue
                print(json.dumps({"source": chunk["source"], "version": chunk["version"],
                                  "locator": locator, "text": chunk["text"]}, ensure_ascii=False, indent=2))
                shown += 1
                if shown >= args.limit:
                    break
        if not shown:
            print("No matching chunks found. Generate the demo files and run ingestion first.")
            return 1
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Inspection stopped: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
