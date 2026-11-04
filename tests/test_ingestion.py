"""Check data loss, source identity, reruns, and failure behavior."""

import json
from pathlib import Path
import tempfile
import unittest

from pypdf import PdfWriter

from enterprise_rag.chunking import split_text
from enterprise_rag.loaders import clean_text, load_pages
from enterprise_rag.pipeline import ingest


class ChunkingTests(unittest.TestCase):
    def test_overlap_and_full_coverage(self):
        text = "one two three four five six seven eight nine ten"
        chunks = split_text(text, max_words=4, overlap_words=1)
        self.assertEqual([c.word_count for c in chunks], [4, 4, 4])
        self.assertEqual([c.word_start for c in chunks], [0, 3, 6])
        self.assertEqual(chunks[-1].word_end, 10)
        for chunk in chunks:
            self.assertEqual(chunk.text, text[chunk.char_start:chunk.char_end])
        self.assertEqual(chunks[0].text.split()[-1], chunks[1].text.split()[0])

    def test_empty_input_and_invalid_settings(self):
        self.assertEqual(split_text(" \n ", 4, 1), [])
        for size, overlap in [(0, 0), (4, -1), (4, 4)]:
            with self.assertRaises(ValueError):
                split_text("text", size, overlap)

    def test_cleaning_keeps_paragraphs_and_unicode(self):
        self.assertEqual(clean_text("Café\x00  rules\r\n\r\n  Next\tline "),
                         "Café rules\n\nNext line")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw, self.out = self.root / "raw", self.root / "processed"
        self.raw.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        path = self.raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def chunks(self):
        return [json.loads(line) for line in (self.out / "chunks.jsonl")
                .read_text(encoding="utf-8").splitlines()]

    def registry(self, records):
        path = self.root / "documents.json"
        path.write_text(json.dumps({"schema_version": 1, "documents": records}), encoding="utf-8")
        return path

    @staticmethod
    def record(version="2026", groups=None):
        return {
            "title": "Reimbursement policy",
            "document_family": "reimbursement-policy",
            "version": version,
            "effective_from": f"{version}-01-01",
            "effective_to": None,
            "access_groups": ["employees"] if groups is None else groups,
        }

    def test_rerun_is_deterministic_without_accumulating_chunks(self):
        self.write("hr/leave.txt", "one two three four five six seven")
        first = ingest(self.raw, self.out, 4, 1)
        first_bytes = (self.out / "chunks.jsonl").read_bytes()
        second = ingest(self.raw, self.out, 4, 1)
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, (self.out / "chunks.jsonl").read_bytes())
        self.assertEqual(first["summary"]["total_chunks"], 2)
        self.assertEqual(self.chunks()[0]["source"], "hr/leave.txt")
        self.assertIsNone(self.chunks()[0]["page"])

    def test_changed_content_changes_identity(self):
        self.write("policy.txt", "employees receive eighteen leave days")
        ingest(self.raw, self.out)
        old_id = self.chunks()[0]["chunk_id"]
        self.write("policy.txt", "employees receive twenty leave days")
        ingest(self.raw, self.out)
        self.assertNotEqual(old_id, self.chunks()[0]["chunk_id"])

    def test_duplicates_are_recorded_and_bad_files_reported(self):
        self.write("a.txt", "Original policy content for the demo")
        self.write("b.txt", "Original policy content for the demo")
        (self.raw / "broken.pdf").write_bytes(b"not a PDF")
        manifest = ingest(self.raw, self.out)
        self.assertEqual(manifest["summary"]["indexed_documents"], 1)
        self.assertEqual(manifest["summary"]["duplicate_documents"], 1)
        self.assertEqual(manifest["summary"]["failed_documents"], 1)
        self.assertEqual(manifest["documents"][1]["duplicate_of"], "a.txt")

    def test_unusable_input_keeps_previous_outputs(self):
        policy = self.write("policy.txt", "Valid policy text")
        ingest(self.raw, self.out)
        previous = (self.out / "chunks.jsonl").read_bytes()
        policy.write_text(" ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Existing outputs were kept"):
            ingest(self.raw, self.out)
        self.assertEqual(previous, (self.out / "chunks.jsonl").read_bytes())

    def test_blank_pdf_is_reported_for_ocr(self):
        pdf = self.raw / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with pdf.open("wb") as handle:
            writer.write(handle)
        pages, warnings = load_pages(pdf.read_bytes(), ".pdf")
        self.assertEqual(pages, [])
        self.assertIn("Page 1", warnings[0])

    def test_input_and_output_must_not_be_nested(self):
        self.write("policy.txt", "Valid policy text")
        for target in (self.raw, self.raw / "output", self.root):
            with self.assertRaisesRegex(ValueError, "non-nested"):
                ingest(self.raw, target)

    def test_versions_and_access_groups_reach_every_chunk(self):
        self.write("old.txt", "Old reimbursement limit is thirty thousand.")
        self.write("new.txt", "New reimbursement limit is forty five thousand.")
        path = self.registry({"old.txt": self.record("2024"),
                              "new.txt": self.record("2026", ["finance", "employees", "finance"])})
        manifest = ingest(self.raw, self.out, metadata_path=path)
        chunks = self.chunks()
        self.assertEqual({c["version"] for c in chunks}, {"2024", "2026"})
        self.assertTrue(all(c["document_family"] == "reimbursement-policy" for c in chunks))
        self.assertTrue(all(c["metadata_registered"] for c in chunks))
        self.assertEqual(chunks[0]["access_groups"], ["employees", "finance"])
        self.assertEqual(manifest["summary"]["unregistered_documents"], 0)

    def test_identical_text_does_not_erase_distinct_versions(self):
        self.write("2024.txt", "An unchanged policy paragraph")
        self.write("2026.txt", "An unchanged policy paragraph")
        path = self.registry({"2024.txt": self.record("2024"), "2026.txt": self.record("2026")})
        manifest = ingest(self.raw, self.out, metadata_path=path)
        self.assertEqual(manifest["summary"]["indexed_documents"], 2)
        self.assertEqual(manifest["summary"]["duplicate_documents"], 0)
        self.assertEqual({c["version"] for c in self.chunks()}, {"2024", "2026"})

    def test_metadata_update_refreshes_payload_without_changing_text_identity(self):
        self.write("policy.txt", "A confidential reimbursement policy paragraph")
        path = self.registry({"policy.txt": self.record(groups=["employees"])})
        ingest(self.raw, self.out, metadata_path=path)
        first = self.chunks()[0]
        path = self.registry({"policy.txt": self.record(groups=["finance"])})
        ingest(self.raw, self.out, metadata_path=path)
        second = self.chunks()[0]
        self.assertEqual(first["chunk_id"], second["chunk_id"])
        self.assertEqual(first["file_sha256"], second["file_sha256"])
        self.assertNotEqual(first["metadata_sha256"], second["metadata_sha256"])
        self.assertEqual(second["access_groups"], ["finance"])

    def test_missing_registry_entry_is_not_indexed(self):
        self.write("known.txt", "A registered policy")
        self.write("unregistered.txt", "A document with no registered permissions")
        path = self.registry({"known.txt": self.record()})
        manifest = ingest(self.raw, self.out, metadata_path=path)
        self.assertEqual(manifest["summary"]["failed_documents"], 1)
        self.assertEqual({c["source"] for c in self.chunks()}, {"known.txt"})

    def test_invalid_registry_preserves_previous_output(self):
        self.write("policy.txt", "A valid policy")
        valid = self.record()
        path = self.registry({"policy.txt": valid})
        ingest(self.raw, self.out, metadata_path=path)
        previous = (self.out / "chunks.jsonl").read_bytes()
        invalid_records = [
            {**valid, "effective_from": "2026-99-99"},
            {**valid, "effective_to": "2025-12-31"},
            {**valid, "access_groups": "employees"},
            {**valid, "version": ""},
        ]
        for invalid in invalid_records:
            path = self.registry({"policy.txt": invalid})
            with self.assertRaises(ValueError):
                ingest(self.raw, self.out, metadata_path=path)
            self.assertEqual(previous, (self.out / "chunks.jsonl").read_bytes())

    def test_unclassified_direct_api_input_has_no_access_groups(self):
        self.write("policy.txt", "An experimental document")
        manifest = ingest(self.raw, self.out)
        chunk = self.chunks()[0]
        self.assertEqual(chunk["access_groups"], [])
        self.assertFalse(chunk["metadata_registered"])
        self.assertEqual(manifest["summary"]["unregistered_documents"], 1)


if __name__ == "__main__":
    unittest.main()
