"""Validate source metadata for later version-aware, permission-filtered retrieval.

This module stores access labels. It does not authenticate users or enforce access.
"""

from datetime import date
import json
from pathlib import Path, PurePosixPath

from .loaders import DocumentReadError


REQUIRED_FIELDS = {"title", "document_family", "version", "effective_from", "access_groups"}
OPTIONAL_FIELDS = {"effective_to"}


def _parse_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a date string in YYYY-MM-DD format.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid date in YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD format.")
    return parsed


def normalize_record(source: str, record: object) -> dict:
    """Validate one metadata record and return its canonical form."""
    relative = PurePosixPath(source)
    if (not source or relative.is_absolute() or ".." in relative.parts
            or "\\" in source or relative.as_posix() != source or source == "."):
        raise ValueError(f"Metadata source must be a normalized relative POSIX path: {source!r}")
    if not isinstance(record, dict):
        raise ValueError(f"Metadata for {source} must be an object.")
    missing = REQUIRED_FIELDS - set(record)
    unknown = set(record) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if missing or unknown:
        raise ValueError(f"Metadata fields for {source}: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for field in ("title", "document_family", "version"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{source}: {field} must be a nonempty string.")
    start = _parse_date(record["effective_from"], f"{source}: effective_from")
    end_value = record.get("effective_to")
    end = _parse_date(end_value, f"{source}: effective_to") if end_value is not None else None
    if end is not None and end < start:
        raise ValueError(f"{source}: effective_to cannot be earlier than effective_from.")
    groups = record["access_groups"]
    if not isinstance(groups, list) or any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError(f"{source}: access_groups must be a list of nonempty strings.")
    return {
        "title": record["title"].strip(),
        "document_family": record["document_family"].strip(),
        "version": record["version"].strip(),
        "effective_from": start.isoformat(),
        "effective_to": end.isoformat() if end else None,
        "access_groups": sorted({group.strip() for group in groups}),
        "metadata_registered": True,
    }


def load_registry(path: Path | None) -> dict[str, dict] | None:
    """Validate the complete registry before changing any output files."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "documents"}:
        raise ValueError("Metadata requires exactly schema_version and documents at the top level.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported metadata schema_version; expected integer 1.")
    if not isinstance(payload["documents"], dict):
        raise ValueError("Metadata documents must be an object keyed by relative source path.")

    registry: dict[str, dict] = {}
    for source, record in payload["documents"].items():
        registry[source] = normalize_record(source, record)
    return registry


def metadata_for_source(source: str, registry: dict[str, dict] | None) -> dict:
    if registry is not None:
        if source not in registry:
            raise DocumentReadError(f"No metadata entry for {source}; register it before indexing.")
        return dict(registry[source])
    # The direct Python API may be used without a registry for ingestion experiments.
    # Such chunks must be excluded from any later user-facing retriever.
    return {
        "title": "Unclassified document",
        "document_family": "unclassified",
        "version": "unversioned",
        "effective_from": None,
        "effective_to": None,
        "access_groups": [],
        "metadata_registered": False,
    }
