"""Regression tests for modules rebuilt after an accidental overwrite (see docs/RECOVERY_NOTES.md)."""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from enterprise_rag.access import Constraints, Principal, payload_for
from enterprise_rag.embeddings import HashingEmbedder
from enterprise_rag.fusion import reciprocal_rank_fusion
from enterprise_rag.pipeline import ingest
from enterprise_rag.vector_store import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]


def chunk(i, groups=("employees",), version="2026", registered=True, **extra):
    return {"chunk_id": f"{i:032x}", "document_id": f"d{i}", "source": f"s{i}.txt", "document_family": "f",
            "version": version, "access_groups": list(groups), "effective_from": "2026-01-01",
            "effective_to": None, "metadata_registered": registered, **extra}


class AccessTests(unittest.TestCase):
    def test_fail_closed_rules(self):
        p = payload_for(chunk(1, groups=("hr",)))
        self.assertFalse(Constraints(frozenset({"employees"})).matches(p))
        self.assertFalse(Constraints(frozenset()).matches(p))
        self.assertTrue(Constraints(frozenset({"hr"})).matches(p))
        self.assertFalse(Constraints(frozenset({"hr"})).matches(payload_for(chunk(1, groups=("hr",), registered=False))))
        self.assertFalse(Constraints(frozenset({"hr"}), as_of=date(2025, 6, 1)).matches(p))
        self.assertTrue(Constraints(frozenset({"hr"}), as_of=date(2026, 6, 1)).matches(p))
        self.assertEqual(Principal("u", frozenset({"a"})).is_admin, False)


class FusionTests(unittest.TestCase):
    def test_rank_based_deterministic_and_duplicate_safe(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["c", "a", "a", "z"]}, k=60)
        self.assertEqual([f.chunk_id for f in fused], ["a", "c", "b", "z"])
        self.assertEqual(fused[0].ranks, {"dense": 1, "bm25": 2})
        self.assertAlmostEqual(fused[0].score, 1 / 61 + 1 / 62)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({}, k=0)


class VectorStoreTests(unittest.TestCase):
    def test_filter_before_rank_and_payload_update(self):
        emb = HashingEmbedder(256)
        store = InMemoryVectorStore(emb.dim)
        texts = ["annual leave days", "vpn access request", "claims within fifteen days"]
        vectors = emb.embed_documents(texts)
        store.upsert([(f"{i:032x}", vectors[i], payload_for(chunk(i, groups=("hr",) if i == 0 else ("employees",))))
                      for i in range(3)])
        query = emb.embed_query("annual leave")
        hits = store.search(query, Constraints(frozenset({"employees"})), 5)
        self.assertNotIn(f"{0:032x}", [cid for cid, _ in hits])
        store.update_document_payload("d0", {"access_groups": ["employees"]})
        self.assertEqual(store.search(query, Constraints(frozenset({"employees"})), 1)[0][0], f"{0:032x}")
        with self.assertRaises(ValueError):
            store.upsert([("x", np.zeros(5, dtype=np.float32), payload_for(chunk(9)))])
        with tempfile.TemporaryDirectory() as d:
            store.save(Path(d))
            clone = InMemoryVectorStore(emb.dim)
            self.assertTrue(clone.load(Path(d), store._payload))
            self.assertEqual(clone.chunk_ids(), store.chunk_ids())


class SemanticChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = ingest(ROOT / "data/raw", Path(self.tmp.name), metadata_path=ROOT / "config/documents.json",
                               strategy="semantic")
        self.chunks = [json.loads(line) for line in (Path(self.tmp.name) / "chunks.jsonl").open(encoding="utf-8")]

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_failures_unique_ids_and_metadata(self):
        self.assertEqual(self.manifest["summary"]["failed_documents"], 0)
        ids = [c["chunk_id"] for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(c["access_groups"] and c["version"] and c["locator"] for c in self.chunks))

    def test_headings_are_context_not_chunks(self):
        headings = {"Annual leave", "Sick leave", "Creating a support ticket", "Appeals"}
        self.assertFalse([c for c in self.chunks if c["text"].strip() in headings])
        leave = [c for c in self.chunks if c["source"] == "leave_policy.txt"]
        self.assertTrue(any("Annual leave" in c["section_path"] for c in leave))

    def test_revisions_stay_separate_and_sections_do_not_leak(self):
        r24 = [c for c in self.chunks if c["source"] == "reimbursement_policy_2024.txt"]
        r26 = [c for c in self.chunks if c["source"] == "reimbursement_policy_2026.txt"]
        self.assertTrue(r24 and r26)
        self.assertTrue(all("45,000" not in c["text"] for c in r24))
        self.assertTrue(any("45,000" in c["text"] for c in r26))
        sick = next(c for c in self.chunks if c["source"] == "leave_policy.txt" and "Sick leave" in c["section_path"])
        self.assertTrue(sick["text"].startswith("Employees receive eight days"))

    def test_table_rows_stay_atomic_with_cell_ranges(self):
        rows = [c for c in self.chunks if c["source"] == "demo/expense_limits.xlsx"]
        self.assertEqual([c["locator"]["cell_range"] for c in rows], ["A2:C2", "A3:C3", "A4:C4"])

    def test_spans_point_into_source_blocks(self):
        for c in (c for c in self.chunks if c["source"].endswith("leave_policy.txt")):
            self.assertTrue(c["char_start"] >= 0 and c["char_end"] > c["char_start"])


if __name__ == "__main__":
    unittest.main()
