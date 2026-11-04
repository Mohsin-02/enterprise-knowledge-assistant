"""Run after patching:  python -m unittest test_llm_fallback   (from the project root)"""

from dataclasses import replace
from datetime import date
from pathlib import Path
import tempfile
import unittest

from enterprise_rag.access import Principal
from enterprise_rag.factory import build_stack
from enterprise_rag.rag import SessionState
from enterprise_rag.settings import Settings

COMPARE = "What changes were made to the employee reimbursement policy between 2024 and 2026?"


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.calls, self.prompts = reply, 0, []

    def generate(self, system, user, *, max_tokens=700):
        self.calls += 1
        self.prompts.append(user)
        return self.reply


class FallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        # force the offline stack so the test is fast and does not need models
        settings = replace(Settings(), embedder="hashing", reranker="lexical", vector_store="memory",
                           processed_dir=tmp / "p", index_dir=tmp / "i", llm_url=None)
        cls.stack = build_stack(settings, today=date(2026, 11, 3), rebuild=True)
        cls.alice = Principal("alice", frozenset({"employees"}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ask(self, reply, question):
        llm = FakeLLM(reply)
        self.stack.rag.llm = llm
        try:
            return self.stack.rag.answer(question, self.alice, SessionState()), llm
        finally:
            self.stack.rag.llm = None

    def test_uncited_model_answer_falls_back_to_quotes(self):
        r, llm = self.ask("Employees get 18 days.", "How many days of paid annual leave do employees get?")
        self.assertEqual(llm.calls, 1)
        self.assertFalse(r.abstained)
        self.assertIn("eighteen", r.answer)
        self.assertTrue(r.citations)
        self.assertTrue(any("quoted passages" in w for w in r.warnings))

    def test_cited_model_answer_is_kept(self):
        r, _ = self.ask("Employees get eighteen days of leave per year. [E1]", "How many days of paid annual leave do employees get?")
        self.assertFalse(r.abstained)
        self.assertTrue(r.answer.startswith("Employees get eighteen days"))
        self.assertFalse(any("quoted passages" in w for w in r.warnings))

    def test_uncited_comparison_falls_back_to_both_versions(self):
        r, _ = self.ask("The limit went up and the deadline got shorter.", COMPARE)
        self.assertFalse(r.abstained)
        for fact in ("30,000", "45,000", "thirty", "fifteen"):
            self.assertIn(fact, r.answer)
        self.assertTrue({"2024", "2026"} <= {c["version"] for c in r.citations})

    def test_insufficient_evidence_still_abstains(self):
        r, _ = self.ask("INSUFFICIENT_EVIDENCE", "How many days of paid annual leave do employees get?")
        self.assertTrue(r.abstained)

    def test_restricted_text_never_reaches_the_model_or_the_answer(self):
        r, llm = self.ask("Form HR-17 must be submitted.", "What form must a manager submit to report misconduct?")
        self.assertTrue(r.abstained)
        self.assertNotIn("HR-17", r.answer)
        for prompt in llm.prompts:  # the model may be called, but only with passages alice may see
            self.assertNotIn("HR-17", prompt)
            self.assertNotIn("misconduct", prompt.lower().split("evidence:")[-1])


if __name__ == "__main__":
    unittest.main()
