"""Retrieval and answer metrics (pure functions, unit-tested) used by evaluate.py."""

from __future__ import annotations

import math
import re

from .textnorm import numbers_in, tokenize

CITE = re.compile(r"\[E(\d+)\]")


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Standard definition: divide by k even when fewer than k passages were returned."""
    return sum(1 for c in ranked[:k] if c in relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(ranked, 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance NDCG."""
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], 1) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _norm(text: str) -> str:
    return text.lower().replace(",", "")


def contains_all(answer: str, needles: list[str]) -> bool:
    haystack = _norm(answer)
    return all(_norm(n) in haystack for n in needles)


def contains_any(answer: str, needles: list[str]) -> bool:
    haystack = _norm(answer)
    return any(_norm(n) in haystack for n in needles)


def line_support(line: str, cited_texts: list[str], min_coverage: float = 0.9) -> bool:
    """Lexical support proxy: every number, and >= `min_coverage` of the content words, of a
    claim line must occur in the passages it cites. It checks that the cited text backs the claim
    lexically; it is NOT an entailment judge and cannot catch a subtle misreading."""
    claim = CITE.sub("", line)
    quoted = re.findall(r'"([^"]+)"', claim)
    # Comparison lines wrap quotes in template words ("Changed. 2024: ..."); judge only the quoted text.
    claim = " ".join(quoted) if quoted else claim.strip("-• \t")
    if not claim:
        return True
    evidence = " ".join(cited_texts)
    if not numbers_in(claim) <= numbers_in(evidence):
        return False
    terms = set(tokenize(claim))
    if not terms:
        return True
    present = set(tokenize(evidence))
    return len(terms & present) / len(terms) >= min_coverage


def answer_faithfulness(answer: str, citations: list[dict]) -> tuple[int, int]:
    """Return (supported_lines, claim_lines) over lines that carry [E#] markers."""
    by_id = {c["id"]: c["text"] for c in citations}
    supported = total = 0
    for line in answer.splitlines():
        ids = [f"E{n}" for n in CITE.findall(line)]
        if not ids:
            continue
        total += 1
        if line_support(line, [by_id[i] for i in ids if i in by_id]):
            supported += 1
    return supported, total
