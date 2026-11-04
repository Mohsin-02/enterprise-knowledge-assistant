"""End-to-end tests of the service layer on the demo corpus (offline stack, temp database)."""

from dataclasses import replace
from datetime import date
from pathlib import Path
import tempfile
import unittest

from enterprise_rag.db import Database
from enterprise_rag.factory import build_stack
from enterprise_rag.service import AccessDenied, KnowledgeService, NotFound
from enterprise_rag.settings import Settings

COMPARE = "What changes were made to the employee reimbursement policy between 2024 and 2026?"
FORM = "What form must a manager submit to report misconduct?"


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        settings = replace(Settings(), database_url=f"sqlite:///{tmp}/app.db", processed_dir=tmp / "processed",
                           index_dir=tmp / "index", upload_dir=tmp / "uploads", rate_limit_per_minute=0)
        cls.db = Database(settings.database_url)
        cls.db.init_schema()
        cls.service = KnowledgeService(build_stack(settings, today=date(2026, 11, 3), rebuild=True), cls.db)
        cls.service.sync_document_records()
        cls.tokens = {u: cls.db.upsert_user(u, u, g, is_admin=(u == "admin")) for u, g in [
            ("alice", ["employees"]), ("bob", ["employees", "finance"]), ("carol", ["employees", "hr"]),
            ("admin", ["employees", "finance", "hr", "it", "security"])]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def who(self, name):
        return self.db.authenticate(self.tokens[name])

    def test_authentication(self):
        self.assertIsNone(self.db.authenticate("wrong"))
        self.assertIsNone(self.db.authenticate(None))
        self.assertEqual(self.who("bob").groups, frozenset({"employees", "finance"}))
        self.assertIsNone(self.db.upsert_user("bob", "Bob", ["employees"]))  # existing user: token unchanged
        self.assertIsNotNone(self.who("bob"))

    def test_headline_comparison_cites_both_revisions(self):
        result = self.service.ask(self.who("alice"), COMPARE)
        self.assertFalse(result["abstained"])
        self.assertEqual(result["analysis"]["versions"], ["2024", "2026"])
        self.assertTrue({"2024", "2026"} <= {c["version"] for c in result["citations"]})
        for fact in ("30,000", "45,000", "thirty calendar days", "fifteen calendar days"):
            self.assertIn(fact, result["answer"])

    def test_followup_keeps_topic_in_session(self):
        first = self.service.ask(self.who("alice"), COMPARE)
        second = self.service.ask(self.who("alice"), "What about the submission deadline?", first["session_id"])
        self.assertTrue(second["analysis"]["followup"])
        self.assertIn("thirty", second["answer"])
        self.assertIn("fifteen", second["answer"])

    def test_restricted_documents_never_leak(self):
        denied = self.service.ask(self.who("alice"), FORM)
        self.assertTrue(denied["abstained"])
        self.assertEqual(denied["citations"], [])
        self.assertNotIn("HR-17", repr(denied))
        self.assertIn("HR-17", self.service.ask(self.who("carol"), FORM)["answer"])

    def test_sessions_are_private(self):
        sid = self.service.ask(self.who("carol"), FORM)["session_id"]
        with self.assertRaises(NotFound):
            self.service.ask(self.who("alice"), "hello", sid)
        with self.assertRaises(NotFound):
            self.service.history(self.who("alice"), sid)
        with self.assertRaises(NotFound):
            self.service.delete_session(self.who("alice"), sid)

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.service.ask(self.who("alice"), "   ")
        with self.assertRaises(ValueError):
            self.service.ask(self.who("alice"), "x" * 5000)

    def test_permission_change_takes_effect_everywhere(self):
        admin = self.who("admin")
        sid = self.service.ask(self.who("bob"), "What is the monthly home internet reimbursement limit?")["session_id"]
        self.assertFalse(self.service.history(self.who("bob"), sid)[1]["redacted"])
        with self.assertRaises(AccessDenied):
            self.service.update_permissions(self.who("alice"), "x", ["employees"])
        family = [d for d in self.service.list_documents(admin) if d["document_family"] == "reimbursement-policy"]
        sheet = [d for d in self.service.list_documents(admin) if d["document_family"] == "expense-limit-table"]
        try:
            for d in family + sheet:
                self.service.update_permissions(admin, d["document_id"], ["hr"])
            self.assertTrue(self.service.history(self.who("bob"), sid)[1]["redacted"])
            again = self.service.ask(self.who("bob"), "What is the monthly home internet reimbursement limit?")
            self.assertTrue(again["abstained"])  # no stale cached answer
        finally:
            for d in family:
                self.service.update_permissions(admin, d["document_id"], ["employees", "finance", "hr"])
            for d in sheet:
                self.service.update_permissions(admin, d["document_id"], ["employees"])

    def test_upload_is_searchable_and_admin_only(self):
        record = {"title": "Travel policy", "document_family": "travel-policy", "version": "2026",
                  "effective_from": "2026-01-01", "access_groups": ["hr"]}
        data = b"Travel policy\n\nEmployees must book flights through the travel desk at least ten days in advance.\n"
        with self.assertRaises(AccessDenied):
            self.service.ingest_upload(self.who("alice"), "travel.txt", data, record)
        with self.assertRaises(ValueError):
            self.service.ingest_upload(self.who("admin"), "travel.txt", data, {"title": "bad"})
        info = self.service.ingest_upload(self.who("admin"), "travel.txt", data, record)
        question = "How many days in advance must flights be booked?"
        self.assertFalse(self.service.ask(self.who("carol"), question)["abstained"])
        self.assertTrue(self.service.ask(self.who("alice"), question)["abstained"])
        self.service.delete_document(self.who("admin"), info["document_id"])
        self.assertTrue(self.service.ask(self.who("carol"), question)["abstained"])

    def test_audit_log_has_no_answer_text(self):
        self.service.ask(self.who("alice"), "How do I request VPN access?")
        events = self.db.audit_events(5)
        self.assertEqual(events[0]["event"], "ask")
        self.assertNotIn("VPN", repr(events[0]["detail"]))


if __name__ == "__main__":
    unittest.main()
