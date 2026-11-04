"""List the evaluation questions that went wrong, and compare the language model with quote-only answers.

Run from the project root, in the same window (same `set ERAG_...` variables) as evaluate.py:
    python show_failures.py                 -> writes eval\\failures.txt and prints it
Takes as long as evaluate.py's answer part (every question is answered twice).
"""

from dataclasses import replace
from pathlib import Path
import tempfile

from evaluate import TODAY, load_questions, run_answers
from enterprise_rag.factory import build_stack
from enterprise_rag.settings import Settings


def bad(row: dict) -> bool:
    lines_ok = row.get("faithful_lines", 0) >= row.get("claim_lines", 0)
    return (not row["correct"]) or (not lines_ok) or row.get("versions_cited") is False \
        or bool(row["leaked"]) or bool(row["forbidden_text"] and row["expected_abstain"])


def main() -> int:
    questions = load_questions(Path("eval/questions.jsonl"))
    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(Settings.from_env(), processed_dir=Path(tmp) / "p", index_dir=Path(tmp) / "i")
        stack = build_stack(settings, today=TODAY, rebuild=True)
        llm = stack.rag.llm
        with_llm = run_answers(stack, questions, None)
        stack.rag.llm = None
        quotes = run_answers(stack, questions, None)
        stack.rag.llm = llm
    out = [f"generator: {getattr(llm, 'name', 'none (quotes only)')}", ""]
    worse = better = 0
    for a, b in zip(with_llm, quotes):
        if not (bad(a) or bad(b)):
            continue
        tag = "LLM WORSE" if bad(a) and not bad(b) else "LLM BETTER" if bad(b) and not bad(a) else "BOTH BAD"
        worse += tag == "LLM WORSE"
        better += tag == "LLM BETTER"
        out += [f"[{tag}] {a['id']} ({a['category']}, {a['split']}) expected_abstain={a['expected_abstain']}",
                f"  Q: {a['question']}",
                f"  LLM   : abstained={a['abstained']} correct={a['correct']} faithful={a.get('faithful_lines')}/{a.get('claim_lines')}"
                f" versions_cited={a.get('versions_cited')} | {a['answer'][:300]!r}",
                f"  QUOTES: abstained={b['abstained']} correct={b['correct']} | {b['answer'][:200]!r}", ""]
    out.append(f"Summary: LLM worse on {worse} turns, better on {better} turns, of {len(with_llm)}.")
    text = "\n".join(out)
    Path("eval").mkdir(exist_ok=True)
    Path("eval/failures.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
