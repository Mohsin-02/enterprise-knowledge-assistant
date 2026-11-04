"""Shared text normalization for BM25, the offline hashing embedder, and support scoring.

Everything here is deterministic and dependency-free. The same function normalizes index
text and query text, which is required for lexical matching to be meaningful.
"""

from functools import lru_cache
import json
from pathlib import Path
import re


STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from into onto about as is are was were be been
being am do does did done doing have has had having it its this that these those there here i me my we our
you your he she they them their his her what which who whom whose when where why how can could should would
will shall may might must not no nor so than too very just also any all each every some such only own same
more most other get got gets please tell us per up out over under again further once
""".split())

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90,
}

_TOKEN = re.compile(r"[a-z0-9]+")
_DIGIT_COMMAS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _stem_once(token: str) -> str:
    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    stripped = False
    if token.endswith("ing") and len(token) > 5:
        token, stripped = token[:-3], True
    elif token.endswith("ed") and len(token) > 4:
        token, stripped = token[:-2], True
    if stripped and len(token) > 3 and token[-1] == token[-2] and token[-1] not in "lsz":
        token = token[:-1]
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def stem(token: str) -> str:
    """A deliberately small suffix stripper. Applied to a fixed point so stems are stable when
    re-tokenized; consistency between index and query matters more than linguistics."""
    for _ in range(3):
        reduced = _stem_once(token)
        if reduced == token:
            break
        token = reduced
    return token


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    text = _DIGIT_COMMAS.sub("", text.lower().replace("₹", " inr "))
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        if raw in NUMBER_WORDS:
            out.append(str(NUMBER_WORDS[raw]))
            continue
        if drop_stopwords and raw in STOPWORDS:
            continue
        out.append(stem(raw))
    return out


@lru_cache(maxsize=1)
def _synonyms() -> dict[str, list[str]]:
    path = Path(__file__).resolve().parent.parent / "config" / "synonyms.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    table: dict[str, list[str]] = {}
    for key, values in raw.items():
        if key.startswith("_") or not isinstance(values, list):
            continue
        table[stem(key.lower())] = [t for value in values for t in tokenize(str(value))]
    return table


def expand_terms(terms: list[str]) -> list[str]:
    """Query-side expansion with the domain thesaurus (lexical fallback only)."""
    table = _synonyms()
    expanded = list(terms)
    for term in terms:
        expanded.extend(table.get(term, []))
    return expanded


def numbers_in(text: str) -> set[str]:
    return {t for t in tokenize(text, drop_stopwords=False) if t.isdigit()}
