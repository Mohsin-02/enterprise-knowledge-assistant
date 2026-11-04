"""Human-readable source references for citations."""

from __future__ import annotations


def describe_locator(locator: dict | None, section: list | None = None) -> str:
    loc = locator or {}
    kind = loc.get("kind")
    parts: list[str] = []
    if kind == "sheet_row":
        parts.append(f"sheet '{loc.get('sheet')}', cells {loc.get('cell_range')}")
    elif kind == "table_row":
        parts.append(f"table {loc.get('table')}, row {loc.get('row')}")
    elif kind == "html_block":
        parts.append("heading path: " + " > ".join(loc.get("section") or []) if loc.get("section") else "page body")
        if loc.get("anchor"):
            parts.append(f"anchor #{loc['anchor']}")
    elif kind == "paragraph":
        parts.append(f"paragraph {loc.get('paragraph')}")
    elif kind == "page":
        parts.append(f"page {loc.get('page')}")
    section = section or loc.get("section")
    if section and kind not in {"html_block"}:
        parts.append("section: " + " > ".join(section))
    return "; ".join(parts) or "whole document"


def format_citation(citation: dict) -> str:
    when = citation.get("effective_from") or "?"
    until = citation.get("effective_to") or "present"
    return (f"[{citation['id']}] {citation['title']} (version {citation['version']}, effective {when} to {until}) "
            f"- {citation['source']} - {describe_locator(citation.get('locator'), citation.get('section'))}")
