"""API tests. Skipped automatically unless fastapi and httpx are installed (not available offline)."""

from dataclasses import replace
from datetime import date
from pathlib import Path
import tempfile
import unittest

try:
    from fastapi.testclient import TestClient
    import api as api_module
except ImportError:  # pragma: no cover
    TestClient = None

from enterprise_rag.db import Database
from enterprise_rag.factory import build_stack
from enterprise_rag.service import KnowledgeService
from enterprise_rag.settings import Settings


@unittest.skipIf(TestClient is None, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        settings = replace(Settings(), database_url=f"sqlite:///{tmp}/a.db", processed_dir=tmp / "p",
                           index_dir=tmp / "i", upload_dir=tmp / "u", rate_limit_per_minute=0)
        db = Database(settings.database_url)
        db.init_schema()
        service = KnowledgeService(build_stack(settings, today=date(2026, 11, 3), rebuild=True), db)
        service.sync_document_records()
        self.alice = db.upsert_user("alice", "Alice", ["employees"])
        self.admin = db.upsert_user("admin", "Admin", ["employees", "hr"], is_admin=True)
        self.client = TestClient(api_module.create_app(service))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_requires_token(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.post("/ask", json={"question": "hi"}).status_code, 401)
        self.assertEqual(self.client.get("/me", headers=self.auth("bad")).status_code, 401)

    def test_ask_and_permissions(self):
        ok = self.client.post("/ask", json={"question": "How many sick leave days per year?"}, headers=self.auth(self.alice))
        self.assertEqual(ok.status_code, 200)
        self.assertIn("eight", ok.json()["answer"])
        denied = self.client.post("/ask", json={"question": "What form must a manager submit to report misconduct?"},
                                  headers=self.auth(self.alice))
        self.assertTrue(denied.json()["abstained"])
        self.assertEqual(self.client.get("/documents", headers=self.auth(self.alice)).status_code, 200)
        self.assertEqual(self.client.delete("/documents/x", headers=self.auth(self.alice)).status_code, 403)

    def test_foreign_session_is_404(self):
        sid = self.client.post("/sessions", headers=self.auth(self.admin)).json()["session_id"]
        self.assertEqual(self.client.get(f"/sessions/{sid}/history", headers=self.auth(self.alice)).status_code, 404)


if __name__ == "__main__":
    unittest.main()
