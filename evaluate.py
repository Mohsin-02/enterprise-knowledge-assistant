"""Evaluate retrieval and answers on eval/questions.jsonl.

  python evaluate.py                    # all metrics, default settings
  python evaluate.py --calibrate        # choose the abstention threshold on the DEV split only
  python evaluate.py --out eval/results.json --write-docs docs/EVALUATION.md

Environment variables (ERAG_EMBEDDER, ERAG_RERANKER, ERAG_LLM_URL, ...) select real models; with the
defaults this runs fully offline with the hashing embedder and the lexical reranker, and the report
says so. Retrieval relevance is defined by (source, phrase) labels, so results survive re-chunking.
"""

import argparse
from dataclasses import replace
from datetime import date
import json
import os
from pathlib import Path
import platform
import tempfile
import time

from enterprise_rag.access import Constraints, Principal
from enterprise_rag.evaluation import (answer_faithfulness, contains_all, contains_any, mean, ndcg_at_k,
                                       percentile, precision_at_k, recall_at_k, reciprocal_rank)
from enterprise_rag.factory import build_stack
from enterprise_rag.rag import AbstentionPolicy, SessionState
from enterprise_rag.retrieval import RetrievalOptions
from enterprise_rag.settings import Settings

TODAY = date(2026, 11, 3)
KS = (1, 3, 5)
CONFIGS = {
    "dense only": RetrievalOptions(mode="dense", rerank=False, final_k=10),
    "BM25 only": RetrievalOptions(mode="bm25", rerank=False, final_k=10),
    "hybrid (RRF)": RetrievalOptions(mode="hybrid", rerank=False, final_k=10),
    "hybrid + rerank": RetrievalOptions(mode="hybrid", rerank=True, final_k=10),
}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relevant_ids(index, labels: list[dict], constraints: Constraints) -> set[str]:
    found: set[str] = set()
    for label in labels:
        matches = {c["chunk_id"] for c in index.chunks.values()
                   if c["source"] == label["source"] and label["contains"].lower() in c["text"].lower()}
        if not matches:
            raise SystemExit(f"Label matches no chunk (fix eval/questions.jsonl): {label}")
        found |= {cid for cid in matches if index.allowed(index.chunks[cid], constraints)}
    return found


def retrieval_constraints(rec: dict) -> Constraints:
    groups = frozenset(rec["groups"])
    if rec.get("versions"):
        return Constraints(groups, versions=frozenset(rec["versions"]))
    return Constraints(groups, as_of=TODAY)


def evaluate_retrieval(stack, questions: list[dict], split: str | None) -> dict:
    rows = [q for q in questions if "relevant" in q and (split is None or q["split"] == split)]
    out = {}
    for name, options in CONFIGS.items():
        metrics = {f"recall@{k}": [] for k in KS}
        metrics.update({f"precision@{k}": [] for k in KS})
        metrics.update({f"ndcg@{k}": [] for k in KS})
        metrics["mrr"], latencies = [], []
        for rec in rows:
            constraints = retrieval_constraints(rec)
            relevant = relevant_ids(stack.index, rec["relevant"], constraints)
            start = time.perf_counter()
            result = stack.retriever.retrieve(rec["question"], constraints, options)
            latencies.append((time.perf_counter() - start) * 1000)
            ranked = [e.chunk["chunk_id"] for e in result.evidence]
            for k in KS:
                metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
                metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
                metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
            metrics["mrr"].append(reciprocal_rank(ranked, relevant))
        out[name] = {**{m: round(mean(v), 3) for m, v in metrics.items()},
                     "latency_ms_p50": round(percentile(latencies, 50), 2),
                     "latency_ms_p95": round(percentile(latencies, 95), 2), "questions": len(rows)}
    return out


def _turn_checks(turn: dict, response, principal: Principal, stack, rec_relevant: set[str]) -> dict:
    expect_abstain = bool(turn.get("expect_abstain"))
    answer = response.answer
    forbidden = any(n.lower() in answer.lower() for n in turn.get("must_not_contain", []))
    if expect_abstain:
        correct = response.abstained and not forbidden
    else:
        correct = (not response.abstained and contains_all(answer, turn.get("answer_contains", []))
                   and (not turn.get("answer_contains_any") or contains_any(answer, turn["answer_contains_any"]))
                   and not forbidden)
    constraints = Constraints(principal.groups)
    leaked = [c["chunk_id"] for c in response.citations + response.evidence
              if not stack.index.allowed(stack.index.chunks[c["chunk_id"]], constraints)]
    cited = {c["chunk_id"] for c in response.citations}
    row = {"abstained": response.abstained, "expected_abstain": expect_abstain, "correct": correct,
           "leaked": len(leaked), "forbidden_text": forbidden}
    if not expect_abstain and not response.abstained:
        if rec_relevant:
            row["citation_precision"] = len(cited & rec_relevant) / len(cited) if cited else 0.0
            row["citation_hit"] = bool(cited & rec_relevant)
        wanted = turn.get("cited_versions")
        if wanted:
            row["versions_cited"] = set(wanted) <= {c["version"] for c in response.citations}
        supported, total = answer_faithfulness(answer, response.citations)
        row["faithful_lines"], row["claim_lines"] = supported, total
    return row


def run_answers(stack, questions: list[dict], split: str | None) -> list[dict]:
    results = []
    for rec in questions:
        if split is not None and rec["split"] != split:
            continue
        principal = Principal(f"eval-{rec['id']}", frozenset(rec["groups"]))
        turns = rec.get("turns") or [rec]
        relevant = (relevant_ids(stack.index, rec["relevant"], Constraints(principal.groups))
                    if rec.get("relevant") else set())
        state = SessionState()
        for turn in turns:
            start = time.perf_counter()
            response = stack.rag.answer(turn["question"], principal, state)
            elapsed = (time.perf_counter() - start) * 1000
            state = response.state
            row = _turn_checks(turn, response, principal, stack, relevant)
            row.update(id=rec["id"], category=rec["category"], split=rec["split"], question=turn["question"],
                       latency_ms=elapsed, answer=response.answer[:300])
            results.append(row)
    return results


def summarize_answers(rows: list[dict]) -> dict:
    if not rows:
        return {}
    answerable = [r for r in rows if not r["expected_abstain"]]
    refusals = [r for r in rows if r["expected_abstain"]]
    cites = [r for r in answerable if "citation_hit" in r]
    versions = [r for r in answerable if "versions_cited" in r]
    lines = sum(r.get("claim_lines", 0) for r in answerable)
    summary = {
        "turns": len(rows),
        "answer_correctness": round(mean([r["correct"] for r in rows]), 3),
        "answerable_correct": round(mean([r["correct"] for r in answerable]), 3),
        "abstention_correct (should-refuse)": round(mean([r["correct"] for r in refusals]), 3),
        "false_abstentions (answerable refused)": sum(1 for r in answerable if r["abstained"]),
        "citation_hit_rate": round(mean([r["citation_hit"] for r in cites]), 3),
        "citation_precision": round(mean([r["citation_precision"] for r in cites]), 3),
        "both_versions_cited (comparisons)": round(mean([r["versions_cited"] for r in versions]), 3) if versions else None,
        "faithfulness_proxy (lexical)": round(sum(r.get("faithful_lines", 0) for r in answerable) / lines, 3) if lines else None,
        "permission_leaks (must be 0)": sum(r["leaked"] + int(r["forbidden_text"] and r["expected_abstain"]) for r in rows),
        "answer_latency_ms_p50": round(percentile([r["latency_ms"] for r in rows], 50), 2),
        "answer_latency_ms_p95": round(percentile([r["latency_ms"] for r in rows], 95), 2),
    }
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    summary["by_category"] = {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())}
    return summary


def calibrate(stack, questions: list[dict]) -> tuple[float, list[tuple[float, float]]]:
    """Pick min_sentence_coverage by DEV correctness only (ties -> the higher, more cautious value)."""
    scored = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        stack.rag.policy = AbstentionPolicy(min_sentence_coverage=threshold)
        scored.append((threshold, mean([r["correct"] for r in run_answers(stack, questions, "dev")])))
    best = max(scored, key=lambda item: (item[1], item[0]))[0]
    return best, scored


def markdown(report: dict) -> str:
    env = report["environment"]
    lines = [f"# Evaluation report", "",
             f"Run: {env['date']} | embedder: `{env['embedder']}` | reranker: `{env['reranker']}` | "
             f"generator: `{env['generator']}`", f"Hardware: {env['hardware']}", "",
             f"Corpus: {env['documents']} documents / {env['chunks']} chunks. Questions: {env['questions']} "
             f"({env['dev']} dev, {env['test']} test).", ""]
    for split, block in report["retrieval"].items():
        lines += [f"## Retrieval ({split})", "", "| configuration | R@1 | R@3 | R@5 | P@3 | P@5 | MRR | NDCG@5 | p50 ms | p95 ms |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, m in block.items():
            lines.append(f"| {name} | {m['recall@1']} | {m['recall@3']} | {m['recall@5']} | {m['precision@3']} | "
                         f"{m['precision@5']} | {m['mrr']} | {m['ndcg@5']} | {m['latency_ms_p50']} | {m['latency_ms_p95']} |")
        lines.append("")
    for split, summary in report["answers"].items():
        lines += [f"## Answers ({split})", ""]
        lines += [f"- {k}: {v}" for k, v in summary.items() if k != "by_category"]
        lines += ["- by category: " + ", ".join(f"{c} {v}" for c, v in summary["by_category"].items()), ""]
    if report.get("calibration"):
        c = report["calibration"]
        lines += ["## Abstention threshold (calibrated on dev only)", "",
                  "| min_sentence_coverage | dev correctness |", "|---|---|"]
        lines += [f"| {t} | {round(s, 3)} |" for t, s in c["sweep"]]
        lines += ["", f"Chosen on dev: {c['chosen']} (the shipped default is {AbstentionPolicy().min_sentence_coverage}).", ""]
        for split, summary in c["answers_at_chosen"].items():
            lines += [f"Answers at the chosen threshold ({split}): correctness {summary['answer_correctness']}, "
                      f"should-refuse correct {summary['abstention_correct (should-refuse)']}, "
                      f"false abstentions {summary['false_abstentions (answerable refused)']}"]
        lines += ["", "Read this as a warning, not a recommendation: with only 20 dev questions the sweep favours "
                  "permissive thresholds, which can make the system answer questions it should refuse on the "
                  "test split. Keep the default unless you calibrate on a much larger, representative labelled set.", ""]
    lines += ["## Caveats", "", *[f"- {c}" for c in report["caveats"]], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-docs", type=Path)
    args = parser.parse_args()
    questions = load_questions(args.questions)
    with tempfile.TemporaryDirectory() as tmp:
        base = Settings.from_env()
        settings = replace(base, processed_dir=Path(tmp) / "processed", index_dir=Path(tmp) / "index")
        stack = build_stack(settings, today=TODAY, rebuild=True)
        calibration = None
        answers_default = {s: summarize_answers(run_answers(stack, questions, s)) for s in ("dev", "test")}
        if args.calibrate:
            chosen, sweep = calibrate(stack, questions)
            stack.rag.policy = AbstentionPolicy(min_sentence_coverage=chosen)
            calibration = {"chosen": chosen, "sweep": sweep,
                           "answers_at_chosen": {s: summarize_answers(run_answers(stack, questions, s))
                                                 for s in ("dev", "test")}}
        report = {
            "environment": {
                "date": date.today().isoformat(), "embedder": stack.embedder.fingerprint,
                "reranker": getattr(stack.retriever.reranker, "name", "none"),
                "generator": getattr(stack.rag.llm, "name", "extractive (no LLM)"),
                "hardware": f"{platform.processor() or platform.machine()}, {os.cpu_count()} CPU(s), "
                            f"Python {platform.python_version()}",
                "documents": len(stack.index.documents()), "chunks": len(stack.index.chunks),
                "questions": len(questions), "dev": sum(q["split"] == "dev" for q in questions),
                "test": sum(q["split"] == "test" for q in questions)},
            "retrieval": {s: evaluate_retrieval(stack, questions, s) for s in ("dev", "test")},
            "answers": answers_default,
            "calibration": calibration,
            "caveats": [
                "Small synthetic corpus (about 30 chunks): retrieval metrics saturate and differences between "
                "configurations are not statistically meaningful. Do not read them as production quality.",
                "With the hashing embedder, 'dense' retrieval is lexical; it does not capture paraphrase the way "
                "BGE does. Re-run with ERAG_EMBEDDER=bge-small (or bge-m3) and ERAG_RERANKER=cross-encoder.",
                "The lexical reranker is a heuristic stand-in for a cross-encoder.",
                "Without an LLM the answers are extractive quotes, so faithfulness_proxy is near-trivial; it is a "
                "lexical support check, not an entailment judge. Inspect answers by hand and, if you add an LLM "
                "judge, treat it as supplementary evidence.",
                "Latency is measured on this machine only and excludes model download/warm-up.",
                "The abstention threshold is chosen on the dev split only (--calibrate). Be aware that a few heuristic "
                "fixes in rag.py (question-filler words, section context in support scoring) were written while "
                "probing questions that also appear in the dev split, so dev scores are optimistic; the test split "
                "was not used to tune anything, and its failures (see by-category counts) are left unfixed on purpose.",
                "Known failure types with the offline stack: vocabulary gaps ('file a claim' vs 'submitted'; "
                "'annual limit' vs 'up to INR 30,000 per calendar year'). A neural embedder, a cross-encoder and an "
                "LLM are expected to help, but that has not been measured here."]}
    text = markdown(report)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    if args.write_docs:
        args.write_docs.write_text(text, encoding="utf-8")
    leaks = sum(s["permission_leaks (must be 0)"] for s in report["answers"].values() if s)
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
