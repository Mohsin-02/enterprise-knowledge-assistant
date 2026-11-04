"""Principals and retrieval constraints.

One `Constraints` object is applied to BOTH retrieval branches before ranking, so unauthorized
passages never become candidates, never reach rank fusion, and never reach a prompt.
Group membership comes from trusted server records (see db.py), never from the request body.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

OPEN_ENDED = 10000000


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset[str]
    is_admin: bool = False


def date_ordinal(value: str | None, default: int) -> int:
    return date.fromisoformat(value).toordinal() if value else default


def payload_for(chunk: dict) -> dict:
    """The filterable fields stored beside every vector and lexical entry."""
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source": chunk["source"],
        "document_family": chunk["document_family"],
        "version": chunk["version"],
        "access_groups": list(chunk["access_groups"]),
        "effective_from_ord": date_ordinal(chunk.get("effective_from"), OPEN_ENDED),
        "effective_to_ord": date_ordinal(chunk.get("effective_to"), OPEN_ENDED),
        "metadata_registered": bool(chunk.get("metadata_registered", False)),
    }


@dataclass(frozen=True)
class Constraints:
    groups: frozenset[str]
    families: frozenset[str] | None = None
    versions: frozenset[str] | None = None
    document_ids: frozenset[str] | None = None
    as_of: date | None = None

    def matches(self, payload: dict) -> bool:
        if not payload.get("metadata_registered"):
            return False
        if not self.groups or not self.groups.intersection(payload["access_groups"]):
            return False
        if self.families is not None and payload["document_family"] not in self.families:
            return False
        if self.versions is not None and payload["version"] not in self.versions:
            return False
        if self.document_ids is not None and payload["document_id"] not in self.document_ids:
            return False
        if self.as_of is not None:
            day = self.as_of.toordinal()
            if not payload["effective_from_ord"] <= day <= payload["effective_to_ord"]:
                return False
        return True

    def cache_key(self) -> tuple:
        def norm(items):
            return None if items is None else tuple(sorted(items))
        return (norm(self.groups), norm(self.families), norm(self.versions), norm(self.document_ids),
                self.as_of.isoformat() if self.as_of else None)
