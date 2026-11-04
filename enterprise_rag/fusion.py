"""Reciprocal Rank Fusion: combines rank positions, never raw scores.

Reconstructed after an accidental overwrite (the original source was not recoverable); written to
the interface `retrieval.py` uses: ``reciprocal_rank_fusion({"dense": [...], "bm25": [...]}, k=60)``
returning objects with ``chunk_id``, ``score`` and per-branch ``ranks``, best first.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fused:
    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: dict[str, list[str]], *, k: int = 60,
                           weights: dict[str, float] | None = None) -> list[Fused]:
    """`rankings` maps a branch name to best-first chunk ids. Ties break deterministically by id."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for branch, ids in rankings.items():
        weight = (weights or {}).get(branch, 1.0)
        seen: set[str] = set()
        position = 0
        for cid in ids:
            if cid in seen:  # a repeated id inside one branch must not count twice
                continue
            seen.add(cid)
            position += 1
            scores[cid] = scores.get(cid, 0.0) + weight / (k + position)
            ranks.setdefault(cid, {})[branch] = position
    return [Fused(cid, scores[cid], ranks[cid]) for cid in sorted(scores, key=lambda c: (-scores[c], c))]
