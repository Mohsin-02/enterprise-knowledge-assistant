"""Token-aware, structure- and similarity-aware chunking that keeps exact source spans.

Reconstructed after an accidental overwrite: the original module was not recoverable, so this is
a new implementation of the interface `pipeline.build_chunks` relies on
(`SemanticOptions`, `semantic_chunk_blocks`). Boundaries, strongest first: section changes
(headings are never chunks of their own), table and sheet rows (atomic, with their first-row
context), paragraph breaks once a chunk has enough content, and similarity valleys between
adjacent sentences (embedder cosine when an embedder is given). Tiny prose fragments are folded
into the following chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from .loaders import ParsedBlock
from .tokens import TokenCounter, split_sentence_spans

_PARA_BREAK = re.compile(r"\n\s*\n")
_ATOMIC = {"table_row", "sheet_row"}


@dataclass(frozen=True)
class SemanticOptions:
    max_tokens: int = 180
    min_tokens: int = 45
    breakpoint_sigma: float = 1.0   # split where adjacent similarity < mean - sigma * std
    min_sentences_for_breakpoints: int = 6


@dataclass
class SemanticChunk:
    text: str
    first_block: int
    locator: dict
    char_start: int          # start offset inside the first block
    char_end: int            # end offset inside the last block
    page: int | None
    contextual_text: str
    section_path: list[str]
    token_count: int
    spans: list[dict] = field(default_factory=list)
    locators: list[dict] = field(default_factory=list)


@dataclass
class _Unit:
    block: int
    start: int
    end: int
    para_start: bool


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    out, last = [], 0
    for match in _PARA_BREAK.finditer(text):
        if text[last:match.start()].strip():
            out.append((last, match.start()))
        last = match.end()
    if text[last:].strip():
        out.append((last, len(text)))
    return out


def _is_heading_line(line: str) -> bool:
    words = line.split()
    return 0 < len(words) <= 10 and not re.search(r"[.!?:;,]$", line.strip())


def _is_structured_heading(block: ParsedBlock, previous_section) -> bool:
    loc = block.locator or {}
    if loc.get("kind") == "html_block":
        return bool(re.fullmatch(r"h[1-6]", loc.get("tag", "")))
    if loc.get("kind") == "paragraph" and loc.get("section"):
        return block.text == loc["section"][-1] and loc["section"] != previous_section
    return False


def _similarities(texts: list[str], embedder) -> np.ndarray:
    if len(texts) < 2:
        return np.zeros(0)
    if embedder is not None:
        vectors = embedder.embed_documents(texts)
        return np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        matrix = TfidfVectorizer().fit_transform(texts)
    except ValueError:
        return np.zeros(len(texts) - 1)
    return np.asarray([matrix[i].multiply(matrix[i + 1]).sum() for i in range(len(texts) - 1)])


def semantic_chunk_blocks(blocks: list[ParsedBlock], *, title: str = "", embedder=None,
                          options: SemanticOptions | None = None, counter: TokenCounter | None = None
                          ) -> list[SemanticChunk]:
    opts = options or SemanticOptions()
    counter = counter or TokenCounter()
    chunks: list[SemanticChunk] = []
    section: list[str] = []
    pending: list[_Unit] = []
    previous_struct_section = None
    first_heading_seen = False

    def context() -> str:
        trail = " > ".join(dict.fromkeys(p for p in [title] + section if p))
        return trail

    def emit(group: list[_Unit]) -> None:
        text = "\n".join(blocks[u.block].text[u.start:u.end] for u in group)
        indices = sorted({u.block for u in group})
        locs = [blocks[i].locator or {} for i in indices]
        loc = dict(locs[0]) if locs else {}
        if section:
            loc["section"] = list(section)
        ctx = context()
        chunks.append(SemanticChunk(
            text=text, first_block=indices[0], locator=loc, char_start=group[0].start, char_end=group[-1].end,
            page=blocks[indices[0]].number, contextual_text=(f"{ctx}\n{text}" if ctx else text),
            section_path=list(section), token_count=counter.count(text),
            spans=[{"block_index": u.block, "char_start": u.start, "char_end": u.end} for u in group],
            locators=locs))

    def flush() -> None:
        nonlocal pending
        if pending:
            for group in _group(pending):
                emit(group)
        pending = []

    def _group(units: list[_Unit]) -> list[list[_Unit]]:
        texts = [blocks[u.block].text[u.start:u.end] for u in units]
        sims = (_similarities(texts, embedder)
                if len(texts) >= opts.min_sentences_for_breakpoints else np.zeros(0))
        cut = sims.mean() - opts.breakpoint_sigma * sims.std() if len(sims) >= 5 else -1.0
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        used = 0
        for i, (unit, text) in enumerate(zip(units, texts)):
            n = counter.count(text)
            boundary = bool(current) and (
                used + n > opts.max_tokens
                or (used >= opts.min_tokens and unit.para_start)
                or (used >= opts.min_tokens and 0 < i <= len(sims) and sims[i - 1] < cut))
            if boundary:
                groups.append(current)
                current, used = [], 0
            current.append(unit)
            used += n
        if current:
            tail_tokens = sum(counter.count(blocks[u.block].text[u.start:u.end]) for u in current)
            if groups and tail_tokens < opts.min_tokens // 2:
                previous = groups[-1]
                joined = sum(counter.count(blocks[u.block].text[u.start:u.end]) for u in previous + current)
                if joined <= int(opts.max_tokens * 1.25):
                    previous.extend(current)
                    return groups
            groups.append(current)
        return groups

    row_totals: dict[tuple, int] = {}
    for block in blocks:
        loc = block.locator or {}
        if loc.get("kind") in _ATOMIC:
            key = (loc["kind"], loc.get("table"), loc.get("sheet"))
            row_totals[key] = row_totals.get(key, 0) + 1

    for index, block in enumerate(blocks):
        loc = block.locator or {}
        kind = loc.get("kind")
        if kind in _ATOMIC:
            flush()
            key = (kind, loc.get("table"), loc.get("sheet"))
            is_context_row = (loc.get("row") == loc.get("context_row")
                              or loc.get("cell_range") == loc.get("context_range"))
            if is_context_row and row_totals[key] > 1:
                continue  # header row is repeated as context inside every body row
            if loc.get("section"):
                section = list(loc["section"])
            emit([_Unit(index, 0, len(block.text), True)])
            continue
        if _is_structured_heading(block, previous_struct_section):
            flush()
            section = list(loc.get("section", [block.text]))
            previous_struct_section = loc.get("section")
            continue
        if kind in {"paragraph", "html_block"} and loc.get("section"):
            previous_struct_section = loc["section"]
            if list(loc["section"]) != section:
                flush()
                section = list(loc["section"])
        for p_start, p_end in _paragraph_spans(block.text):
            paragraph = block.text[p_start:p_end]
            body_start = p_start
            if kind in {"text", "page"}:  # infer headings in flat text
                first, _, rest = paragraph.partition("\n")
                if _is_heading_line(first) and (rest.strip() or len(paragraph.split()) <= 10):
                    flush()
                    heading = first.strip().lstrip("#").strip()
                    if not first_heading_seen:
                        first_heading_seen, section = True, []
                        title = title or heading
                    else:
                        section = [heading]
                    body_start = p_start + len(first) + 1 if rest else p_end
            first_unit = True
            for s_start, s_end in split_sentence_spans(block.text[body_start:p_end]):
                pending.append(_Unit(index, body_start + s_start, body_start + s_end, first_unit))
                first_unit = False
    flush()
    return _fold_tiny(chunks, opts, counter)


def _fold_tiny(chunks: list[SemanticChunk], opts: SemanticOptions, counter: TokenCounter) -> list[SemanticChunk]:
    out: list[SemanticChunk] = []
    i = 0
    prose = lambda c: c.locator.get("kind") not in _ATOMIC
    while i < len(chunks):
        cur = chunks[i]
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        if (nxt is not None and cur.token_count < opts.min_tokens // 2 and prose(cur) and prose(nxt)
                and (cur.section_path == nxt.section_path or not cur.section_path)
                and cur.token_count + nxt.token_count <= opts.max_tokens and cur.spans[-1]["block_index"] <= nxt.first_block):
            text = cur.text + "\n" + nxt.text
            out.append(SemanticChunk(
                text=text, first_block=cur.first_block, locator=cur.locator, char_start=cur.char_start,
                char_end=nxt.char_end, page=cur.page,
                contextual_text=nxt.contextual_text.replace(nxt.text, text), section_path=nxt.section_path,
                token_count=counter.count(text), spans=cur.spans + nxt.spans,
                locators=cur.locators + [l for l in nxt.locators if l not in cur.locators]))
            i += 2
        else:
            out.append(cur)
            i += 1
    return out
