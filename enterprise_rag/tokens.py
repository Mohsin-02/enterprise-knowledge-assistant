"""Sentence splitting and token counting for token-aware chunking."""

import re


_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "no", "approx", "inc", "ltd", "co",
                  "fig", "sec", "st", "cf"}
_BOUNDARY = re.compile(r"[.!?][\"')\]]*(?=\s+[\"'(\[]?[A-Z0-9])")
_LIST_ITEM = re.compile(r"\s*(?:[-*\u2022]|\d+[.)])\s+")


def _soften_newlines(text: str) -> str:
    """Join hard-wrapped lines with a space (same length, so offsets stay valid).

    A newline stays a hard boundary when it ends a sentence, introduces a list item, or follows a
    colon-terminated lead-in or a short heading-like line.
    """
    chars = list(text)
    lines = text.split("\n")
    position = 0
    for index, line in enumerate(lines[:-1]):
        position += len(line)
        following = lines[index + 1]
        stripped = line.rstrip()
        hard = (not stripped or not following.strip() or stripped.endswith((".", "!", "?", ":", ";"))
                or _LIST_ITEM.match(following) or len(stripped.split()) <= 8 and not stripped.endswith(","))
        if not hard:
            chars[position] = " "
        position += 1
    return "".join(chars)


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentences; list items and short lines stand alone."""
    norm = _soften_newlines(text)
    spans: list[tuple[int, int]] = []
    position = 0
    for line in norm.split("\n"):
        line_start = position
        position += len(line) + 1
        if not line.strip():
            continue
        start = 0
        for match in _BOUNDARY.finditer(line):
            head = line[start:match.start() + 1].split()
            last = head[-1].rstrip(".").lower() if head else ""
            if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha() and head[-1][0].isupper()):
                continue
            end = match.end()
            _push(spans, line_start, line, start, end)
            start = end
        _push(spans, line_start, line, start, len(line))
    return spans


def _push(spans: list, base: int, line: str, start: int, end: int) -> None:
    segment = line[start:end]
    stripped = segment.strip()
    if not stripped:
        return
    lead = len(segment) - len(segment.lstrip())
    trail = len(segment) - len(segment.rstrip())
    spans.append((base + start + lead, base + end - trail))


class TokenCounter:
    """Approximate WordPiece-style counts without downloading a tokenizer."""

    name = "approx-wordpiece-v1"

    def count(self, text: str) -> int:
        total = 0
        for match in re.finditer(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", text):
            token = match.group()
            total += 1 + (len(token) - 1) // 7 if token[0].isalnum() else 1
        return total


class HFTokenCounter(TokenCounter):
    """Exact counts for the embedding model's tokenizer (needs the `tokenizers` package)."""

    def __init__(self, tokenizer_name: str):
        from tokenizers import Tokenizer  # optional dependency

        self._tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        self.name = f"hf:{tokenizer_name}"

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
