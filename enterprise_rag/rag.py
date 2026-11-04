"""Question understanding, version-aware retrieval, grounded answers, citation validation, abstention.

Design rules (see docs/ARCHITECTURE.md):
* Every retrieval carries the caller's Constraints; nothing outside them reaches a prompt.
* A comparison retrieves each requested revision separately, so neither side can crowd out the other.
  A missing revision is reported as missing; it is never silently replaced by another revision and
  "no change" is never claimed when one side is absent. A revision that does not exist and one the user
  may not access produce the same message.
* Citations are [E#] markers validated against the evidence actually shown to the model.
* Conversational memory keeps only scope (family, revisions, comparison flag, last standalone
  question), never passage text, and is re-validated against current permissions on each turn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import re

from .access import Constraints, Principal
from .index_manager import IndexManager, embedding_text
from .llm import LLM
from .retrieval import Evidence, HybridRetriever, RetrievalOptions
from .textnorm import expand_terms, numbers_in, tokenize
from .tokens import split_sentence_spans

NO_EVIDENCE = ("I could not find supporting information for that in the documents you have access to, "
               "so I won't guess.")
CITE = re.compile(r"\[E(\d+)\]")
COMPARE_CUES = re.compile(r"\b(chang\w*|differ\w*|compar\w*|vs\.?|versus|updat\w*|modif\w*|revis\w*|new)\b", re.I)
FOLLOWUP_CUES = re.compile(r"^(what|how) about\b|^and\b|^also\b|^(what|how) (is|are) (it|that|they|those)\b|\b(it|that|those|they|them|same)\b", re.I)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
_GENERIC_WORDS = {"change", "changes", "changed", "difference", "differences", "differ", "compare", "compared",
                  "comparison", "policy", "policies", "employee", "employees", "made", "update", "updated", "updates",
                  "modified", "modification", "revise", "revised", "revision", "revisions", "new", "version",
                  "versions", "between", "versus", "vs", "latest", "current", "previous", "old", "summary",
                  "summarize", "all", "detail", "details", "what", "overall"}
# Normalised with the SAME tokenizer as questions, so stems always agree (e.g. "made" -> "mad").
GENERIC_FOCUS = frozenset({t for w in _GENERIC_WORDS for t in tokenize(w)} | _GENERIC_WORDS)
# Question filler that never appears in policy prose and must not count as "unsupported" terms.
QUESTION_NOISE = frozenset(t for w in ("many much give need needs know explain describe list exactly happen happens "
                                       "rules rule").split() for t in tokenize(w))


@dataclass
class SessionState:
    family: str | None = None
    versions: list[str] = field(default_factory=list)
    comparison: bool = False
    last_standalone: str = ""
    focus_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SessionState":
        data = data or {}
        return cls(data.get("family"), list(data.get("versions", [])), bool(data.get("comparison", False)),
                   data.get("last_standalone", ""), list(data.get("focus_terms", [])))


@dataclass
class Analysis:
    standalone: str
    retrieval_query: str
    versions: list[str]
    family: str | None
    comparison: bool
    followup: bool
    focus_terms: list[str]
    unresolved_versions: list[str] = field(default_factory=list)
    as_of: str | None = None


@dataclass
class RAGResponse:
    answer: str
    abstained: bool
    citations: list[dict]
    evidence: list[dict]
    analysis: dict
    diagnostics: dict
    warnings: list[str]
    state: SessionState

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.to_dict()
        return data


@dataclass
class AbstentionPolicy:
    """Thresholds are embedder-specific. Calibrate with `python evaluate.py --calibrate`."""
    min_sentence_coverage: float = 0.5   # idf-weighted share of question terms found in the best sentence
    min_dense_top: float = 0.0           # abstain on LLM path only if coverage AND dense score are both low


class RAGPipeline:
    def __init__(self, index: IndexManager, retriever: HybridRetriever, llm: LLM | None = None, *,
                 options: RetrievalOptions | None = None, policy: AbstentionPolicy | None = None,
                 today: date | None = None, max_compare_chunks: int = 12):
        self.index, self.retriever, self.llm = index, retriever, llm
        self.options = options or RetrievalOptions()
        self.policy = policy or AbstentionPolicy()
        self._today = today
        self.max_compare_chunks = max_compare_chunks

    @property
    def today(self) -> date:
        return self._today or date.today()

    # ---- catalogue of what THIS user may see ---------------------------------------------------
    def _catalog(self, principal: Principal) -> dict[str, dict]:
        families: dict[str, dict] = {}
        for doc in self.index.documents(Constraints(principal.groups)):
            entry = families.setdefault(doc["document_family"], {"title": doc["title"], "revisions": {}})
            entry["revisions"][doc["version"]] = (doc["effective_from"], doc["effective_to"])
        return families

    @staticmethod
    def _family_terms(family: str, title: str) -> set[str]:
        generic = {"polic", "guide", "procedur", "tabl", "handbook"}
        return {t for t in tokenize(family.replace("-", " ")) if t not in generic}

    def _detect_family(self, question: str, catalog: dict[str, dict]) -> str | None:
        q_terms = set(tokenize(question))
        best, best_score = None, 0
        for family, entry in catalog.items():
            wanted = self._family_terms(family, entry["title"])
            if wanted and wanted <= q_terms and len(wanted) > best_score:
                best, best_score = family, len(wanted)
        return best

    @staticmethod
    def _resolve_version(token: str, revisions: dict[str, tuple]) -> str | None:
        if token in revisions:
            return token
        if YEAR.fullmatch(token):
            midyear = date(int(token), 6, 30).isoformat()
            for version, (start, end) in revisions.items():
                if start and start <= midyear and (end is None or midyear <= end):
                    return version
        return None

    # ---- understanding -------------------------------------------------------------------------
    def analyze(self, question: str, principal: Principal, state: SessionState) -> Analysis:
        catalog = self._catalog(principal)
        question = question.strip()
        years = [m.group(0) for m in YEAR.finditer(question)]
        family = self._detect_family(question, catalog)
        q_terms = [t for t in tokenize(question) if not YEAR.fullmatch(t)]
        content = [t for t in q_terms if t not in GENERIC_FOCUS]
        followup = bool(state.last_standalone) and (
            len(question.split()) <= 8 and bool(FOLLOWUP_CUES.search(question)) or not content or not family and len(content) <= 2)
        carried_family = state.family if state.family in catalog else None  # permissions re-checked
        if followup:
            family = family or carried_family
        wants_compare = bool(COMPARE_CUES.search(question)) and (family is not None or len(years) >= 2)
        if followup and not wants_compare and not years and state.comparison:
            wants_compare = True
        requested = list(dict.fromkeys(years)) or (list(state.versions) if followup else [])
        if family is None and wants_compare:
            family = carried_family
        unresolved: list[str] = []
        versions: list[str] = []
        if family and requested:
            revisions = catalog[family]["revisions"]
            for token in requested:
                resolved = self._resolve_version(token, revisions)
                (versions if resolved else unresolved).append(resolved or token)
        elif requested:
            versions = requested  # exact labels, filtered by constraints at retrieval time
        if wants_compare and family and not requested:
            ordered = sorted(catalog[family]["revisions"].items(), key=lambda kv: kv[1][0] or "")
            versions = [v for v, _ in ordered[:1] + ordered[-1:]] if len(ordered) > 1 else [v for v, _ in ordered]
        elif wants_compare and family and len(requested) == 1 and re.search(r"\b(since|from)\b", question, re.I):
            latest = sorted(catalog[family]["revisions"].items(), key=lambda kv: kv[1][0] or "")[-1][0]
            versions = list(dict.fromkeys(versions + [latest]))
        versions = list(dict.fromkeys(versions))
        if wants_compare and len(versions) + len(unresolved) < 2 and not (family and len(catalog[family]["revisions"]) < 2):
            pass
        family_words = self._family_terms(family, catalog[family]["title"]) if family else set()
        focus = [t for t in content if t not in family_words]
        if followup and not focus:  # e.g. "And in 2024?" keeps the previous topic
            focus = list(state.focus_terms)
        title_terms = " ".join(sorted(family_words)) if family else ""
        retrieval_query = " ".join(filter(None, [title_terms, " ".join(focus) if focus else question]))
        if not focus:
            retrieval_query = " ".join(filter(None, [title_terms, question]))
        standalone = question if not followup else f"{question} (topic: {state.last_standalone})"
        as_of = None if (versions or unresolved or wants_compare) else self.today.isoformat()
        return Analysis(standalone, retrieval_query, versions, family, wants_compare, followup,
                        focus, unresolved, as_of)

    # ---- retrieval -----------------------------------------------------------------------------
    def _constraints(self, principal: Principal, analysis: Analysis, **overrides) -> Constraints:
        base = dict(groups=principal.groups,
                    as_of=date.fromisoformat(analysis.as_of) if analysis.as_of else None)
        if analysis.versions and not analysis.comparison:
            base["versions"] = frozenset(analysis.versions)
        if analysis.family and (analysis.comparison or analysis.versions):
            base["families"] = frozenset([analysis.family])
        base.update(overrides)
        return Constraints(**base)

    # ---- answering -----------------------------------------------------------------------------
    def answer(self, question: str, principal: Principal, state: SessionState | None = None,
               options: RetrievalOptions | None = None) -> RAGResponse:
        state = state or SessionState()
        analysis = self.analyze(question, principal, state)
        opts = options or self.options
        warnings: list[str] = []
        if analysis.comparison and analysis.family:
            return self._answer_comparison(analysis, principal, state, opts, warnings)
        constraints = self._constraints(principal, analysis)
        result = self.retriever.retrieve(analysis.retrieval_query, constraints, opts)
        evidence = result.evidence
        weights = self.index.lexical.term_weights(expand_terms(tokenize(analysis.retrieval_query)), constraints)
        if not evidence:
            return self._abstain(analysis, state, result.diagnostics, warnings, evidence)
        if self.llm is None:
            sentences = self._best_sentences(analysis.retrieval_query, evidence, weights)
            if not sentences:
                return self._abstain(analysis, state, result.diagnostics, warnings, evidence)
            text = "\n".join(f"- {s} [E{n}]" for s, n in sentences)
        else:
            best = max((self._coverage(analysis.retrieval_query, e.chunk["text"], weights) for e in evidence), default=0)
            if best < self.policy.min_sentence_coverage and result.dense_top < self.policy.min_dense_top:
                return self._abstain(analysis, state, result.diagnostics, warnings, evidence)
            text = self._llm_answer(question, analysis, evidence)
            if text.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
                return self._abstain(analysis, state, result.diagnostics, warnings, evidence)
        fallback = None
        if self.llm is not None:
            def fallback():
                picked = self._best_sentences(analysis.retrieval_query, evidence, weights)
                return "\n".join(f"- {s} [E{n}]" for s, n in picked) if picked else None
        return self._finish(text, analysis, state, evidence, result.diagnostics, warnings, fallback)

    def _abstain(self, analysis, state, diagnostics, warnings, evidence) -> RAGResponse:
        return RAGResponse(NO_EVIDENCE, True, [], [], asdict(analysis), diagnostics, warnings,
                           self._next_state(analysis, state, answered=False))

    def _next_state(self, analysis: Analysis, state: SessionState, answered: bool) -> SessionState:
        if not answered:
            return state
        return SessionState(analysis.family or state.family, analysis.versions or (state.versions if analysis.followup else []),
                            analysis.comparison, analysis.standalone if not analysis.followup else state.last_standalone,
                            analysis.focus_terms or state.focus_terms)

    def _finish(self, text, analysis, state, evidence, diagnostics, warnings, fallback=None) -> RAGResponse:
        valid, text, problems = validate_citations(text, len(evidence))
        if not valid and fallback is not None:
            alternative = fallback()
            if alternative:
                ok, alt_text, alt_problems = validate_citations(alternative, len(evidence))
                if ok:
                    text, valid, problems = alt_text, True, alt_problems + [
                        "The language model's answer had no valid citations, so the quoted passages are shown instead."]
        warnings = warnings + problems
        if not valid:
            return RAGResponse(NO_EVIDENCE, True, [], [], asdict(analysis), diagnostics,
                               warnings + ["Answer discarded: it contained no valid citation."], state)
        cited = sorted({int(n) for n in CITE.findall(text)})
        citations = [{"id": f"E{n}", **evidence[n - 1].public()} for n in cited]
        shown = [{"id": f"E{i}", **e.public()} for i, e in enumerate(evidence, 1)]
        return RAGResponse(text, False, citations, shown, asdict(analysis), diagnostics, warnings,
                           self._next_state(analysis, state, answered=True))

    # ---- sentence-level support scoring (idf-weighted) ----------------------------------------
    @staticmethod
    def _coverage(query: str, sentence: str, weights: dict[str, float]) -> float:
        terms = [t for t in tokenize(query) if t not in QUESTION_NOISE]
        if not terms:
            return 0.0
        present = set(tokenize(sentence))
        total = got = 0.0
        for term in set(terms):
            weight = weights.get(term, 0.0) or max(weights.values(), default=1.0)
            total += weight
            options = {term, *expand_terms([term])[1:]}
            if options & present:
                got += weight
        return got / total if total else 0.0

    def _best_sentences(self, query: str, evidence: list[Evidence], weights: dict[str, float],
                        limit: int = 3) -> list[tuple[str, int]]:
        scored = []
        for n, e in enumerate(evidence, 1):
            text = e.chunk["text"]
            context = " ".join([e.chunk.get("title", ""), *(e.chunk.get("section_path") or [])])
            for s, en in split_sentence_spans(text):
                sentence = text[s:en]
                score = self._coverage(query, f"{sentence} {context}", weights)
                if score >= self.policy.min_sentence_coverage:
                    scored.append((score - 0.01 * n, sentence, n))
        scored.sort(key=lambda item: -item[0])
        if not scored:
            return []
        top = scored[0][0]
        chosen = [item for item in scored if item[0] >= max(top * 0.7, self.policy.min_sentence_coverage)][:limit]
        return [(sentence, n) for _, sentence, n in chosen]

    # ---- comparison ----------------------------------------------------------------------------
    def _revision_evidence(self, principal: Principal, analysis: Analysis, version: str,
                           opts: RetrievalOptions, generic: bool) -> tuple[list[Evidence], dict]:
        constraints = Constraints(principal.groups, families=frozenset([analysis.family]),
                                  versions=frozenset([version]))
        if generic:
            chunks = self.index.chunks_for(constraints, self.max_compare_chunks + 1)
            if chunks and len(chunks) <= self.max_compare_chunks:
                return [Evidence(c, 1.0) for c in chunks], {"mode": "all-chunks"}
        result = self.retriever.retrieve(analysis.retrieval_query, constraints,
                                         RetrievalOptions(**{**asdict(opts), "final_k": 4}))
        return result.evidence, result.diagnostics

    def _answer_comparison(self, analysis, principal, state, opts, warnings) -> RAGResponse:
        versions = list(analysis.versions)
        generic = not [t for t in analysis.focus_terms if t not in GENERIC_FOCUS]
        notes = [f"No accessible revision labelled {v!r} of this document was found." for v in analysis.unresolved_versions]
        if len(versions) < 2 and not notes:
            notes.append("Only one accessible revision of this document exists, so there is nothing to compare it with.")
        per_version: dict[str, list[Evidence]] = {}
        diagnostics: dict = {"per_revision": {}}
        for v in versions:
            found, diag = self._revision_evidence(principal, analysis, v, opts, generic)
            diagnostics["per_revision"][v] = diag
            if found:
                per_version[v] = found
            else:
                notes.append(f"No accessible revision labelled {v!r} of this document was found.")
        usable = list(per_version)
        if len(usable) < 2:
            if not usable:
                return self._abstain(analysis, state, diagnostics, notes, [])
            only = usable[0]
            flat = per_version[only]
            weights = self.index.lexical.term_weights(expand_terms(tokenize(analysis.retrieval_query)),
                                                      Constraints(principal.groups, families=frozenset([analysis.family])))
            picked = self._best_sentences(analysis.retrieval_query, flat, weights)
            body = "\n".join(f"- {s} [E{n}]" for s, n in picked) or ""
            text = ("I can't compare the revisions because only one was available. " + " ".join(notes)
                    + (f"\nWhat the {only} revision says:\n{body}" if body else ""))
            if not body:
                return self._abstain(analysis, state, diagnostics, notes, flat)
            return self._finish(text, analysis, state, flat, diagnostics, notes)
        evidence = [e for v in usable for e in per_version[v]]
        index_of = {id(e): i for i, e in enumerate(evidence, 1)}
        if self.llm is None:
            text = self._extractive_compare(analysis, usable, per_version, index_of, generic, principal)
            if text is None:
                return self._abstain(analysis, state, diagnostics, notes, evidence)
        else:
            text = self._llm_answer(analysis.standalone, analysis, evidence, comparison=usable)
            if text.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
                return self._abstain(analysis, state, diagnostics, notes, evidence)
        if notes:
            text += "\nNote: " + " ".join(notes)
        fallback = None
        if self.llm is not None:
            def fallback():
                alternative = self._extractive_compare(analysis, usable, per_version, index_of, generic, principal)
                if alternative and notes:
                    alternative += "\nNote: " + " ".join(notes)
                return alternative
        return self._finish(text, analysis, state, evidence, diagnostics, notes, fallback)

    def _extractive_compare(self, analysis, versions, per_version, index_of, generic, principal) -> str | None:
        older, newer = versions[0], versions[-1]
        def sentences(version):
            out = []
            for e in per_version[version]:
                for s, en in split_sentence_spans(e.chunk["text"]):
                    sentence = e.chunk["text"][s:en]
                    if re.match(r"^(version\b|synthetic training)", sentence, re.I):
                        continue
                    out.append((sentence, index_of[id(e)]))
            return out
        old_s, new_s = sentences(older), sentences(newer)
        constraints = Constraints(principal.groups, families=frozenset([analysis.family]))
        weights = self.index.lexical.term_weights(expand_terms(tokenize(analysis.retrieval_query)), constraints)
        if not generic:
            relevant = lambda item: self._coverage(analysis.retrieval_query, item[0], weights) >= self.policy.min_sentence_coverage
            old_s, new_s = [s for s in old_s if relevant(s)], [s for s in new_s if relevant(s)]
            if not old_s and not new_s:
                return None
        def terms(sentence): return {t for t in tokenize(sentence) if not t.isdigit()}
        changed, added, removed, unchanged = [], [], [], 0
        remaining_old = list(old_s)
        remaining_new = []
        for sentence, n in new_s:  # exact matches first
            hit = next((o for o in remaining_old if tokenize(o[0], drop_stopwords=False) == tokenize(sentence, drop_stopwords=False)), None)
            if hit:
                remaining_old.remove(hit)
                unchanged += 1
            else:
                remaining_new.append((sentence, n))
        pairs = []
        for si, (sentence, n) in enumerate(remaining_new):
            a = terms(sentence)
            for oi, (osent, on) in enumerate(remaining_old):
                b = terms(osent)
                shared = len(a & b)
                if shared >= 2:
                    pairs.append((shared / (min(len(a), len(b)) or 1), shared / (len(a | b) or 1), si, oi))
        used_new, used_old = set(), set()
        for coverage, jaccard, si, oi in sorted(pairs, reverse=True):
            if coverage < 0.5 or si in used_new or oi in used_old:
                continue
            used_new.add(si)
            used_old.add(oi)
            changed.append((remaining_old[oi], remaining_new[si]))
        added = [item for i, item in enumerate(remaining_new) if i not in used_new]
        removed = [item for i, item in enumerate(remaining_old) if i not in used_old]
        if not (changed or added or removed):
            return (f"Among the retrieved passages, the {older} and {newer} revisions state the same rules"
                    f" ({unchanged} matching sentences). [E{index_of[id(per_version[older][0])]}] "
                    f"[E{index_of[id(per_version[newer][0])]}]")
        lines = [f"Comparison of the {older} and {newer} revisions (based on retrieved passages):"]
        for (osent, on), (nsent, nn) in changed:
            lines.append(f"- Changed. {older}: \"{osent}\" [E{on}] {newer}: \"{nsent}\" [E{nn}]")
        for sentence, n in added:
            lines.append(f"- In {newer} only (among retrieved passages): \"{sentence}\" [E{n}]")
        for sentence, n in removed:
            lines.append(f"- In {older} only (among retrieved passages): \"{sentence}\" [E{n}]")
        if unchanged:
            lines.append(f"- {unchanged} other sentence(s) are identical in both revisions.")
        return "\n".join(lines)

    # ---- LLM path ------------------------------------------------------------------------------
    SYSTEM = ("You answer questions about company documents using ONLY the numbered evidence supplied. "
              "Cite every factual statement with its evidence tag such as [E2]. Use only tags that appear in the "
              "evidence. If the evidence does not answer the question, reply exactly: INSUFFICIENT_EVIDENCE. "
              "Never use outside knowledge. Quote numbers and dates exactly. Each evidence item names its "
              "document version; never present one version's rule as another's.")

    def _llm_answer(self, question: str, analysis: Analysis, evidence: list[Evidence],
                    comparison: list[str] | None = None) -> str:
        blocks = []
        for i, e in enumerate(evidence, 1):
            c = e.chunk
            blocks.append(f"[E{i}] {c['title']} | version {c['version']} | effective {c.get('effective_from')}"
                          f" to {c.get('effective_to') or 'present'} | {c['source']}\n{embedding_text(c)}")
        task = ("Compare the revisions " + " and ".join(comparison) + ": list what changed, what was added, what "
                "was removed, citing both sides. Do not claim that nothing changed unless both sides are shown.\n"
                if comparison else "")
        user = f"Question: {question}\n{task}\nEvidence:\n" + "\n\n".join(blocks)
        return self.llm.generate(self.SYSTEM, user)


def validate_citations(text: str, evidence_count: int, strict: bool = True) -> tuple[bool, str, list[str]]:
    """Remove references to nonexistent evidence and (strict) uncited factual lines."""
    problems: list[str] = []
    def fix(match):
        n = int(match.group(1))
        if 1 <= n <= evidence_count:
            return match.group(0)
        problems.append(f"Removed invalid citation [E{n}].")
        return ""
    text = CITE.sub(fix, text)
    if strict:
        kept = []
        for line in text.split("\n"):
            stripped = line.strip()
            is_claim = bool(stripped) and not stripped.lower().startswith(("note:", "comparison of", "i can't compare",
                                                                            "what the")) and len(stripped.split()) > 5
            if is_claim and not CITE.search(stripped):
                problems.append(f"Removed uncited statement: {stripped[:60]}")
                continue
            kept.append(line)
        text = "\n".join(kept)
    return bool(CITE.search(text)), text.strip(), problems
