"""Exercise real in-memory DOCX/XLSX files and HTML, including source fidelity."""

from datetime import date
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from docx import Document
from openpyxl import Workbook

from enterprise_rag.loaders import DocumentReadError, load_document
from enterprise_rag.pipeline import ingest


def docx_bytes(document):
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def xlsx_bytes(workbook):
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


class DOCXTests(unittest.TestCase):
    def test_body_order_headings_and_table_locations(self):
        document = Document()
        document.add_heading("Expenses", level=1)
        document.add_paragraph("Submit within fifteen days.")
        table = document.add_table(rows=2, cols=2)
        for cell, text in zip(table.rows[0].cells, ["Category", "Limit"]):
            cell.text = text
        for cell, text in zip(table.rows[1].cells, ["Internet", "1500"]):
            cell.text = text
        document.add_paragraph("Finance reviews the claim.")
        blocks, warnings = load_document(docx_bytes(document), ".docx")
        self.assertEqual(len(blocks), 5)
        self.assertEqual(blocks[1].locator["paragraph"], 2)
        self.assertEqual(blocks[3].locator["table"], 1)
        self.assertEqual(blocks[3].locator["row"], 2)
        self.assertEqual(blocks[3].locator["section"], ["Expenses"])
        self.assertIn("1500", blocks[3].text)
        self.assertTrue(blocks[-1].text.startswith("Finance"))
        self.assertTrue(all(block.number is None for block in blocks))
        self.assertEqual(warnings, [])

    def test_identical_paragraphs_have_distinct_chunk_ids(self):
        document = Document()
        document.add_paragraph("The same sentence.")
        document.add_paragraph("The same sentence.")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "repeat.docx").write_bytes(docx_bytes(document))
            ingest(raw, root / "processed")
            chunks = [json.loads(line) for line in (root / "processed/chunks.jsonl").read_text().splitlines()]
            self.assertEqual(len(chunks), 2)
            self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), 2)
            self.assertEqual([chunk["locator"]["paragraph"] for chunk in chunks], [1, 2])

    def test_broken_docx_is_an_actionable_document_error(self):
        with self.assertRaises(DocumentReadError):
            load_document(b"not a docx", ".docx")


class HTMLTests(unittest.TestCase):
    def test_anchors_unicode_and_noncontent_removal(self):
        markup = b'''<html><head><title>Ignore title</title></head><body><nav>Ignore nav</nav>
<h1 id="policy">Policy</h1><h2 id="deadline">Deadline</h2><p>Submit in <b>15 days</b> &amp; keep receipts.</p>
<script>ignore_script</script><p hidden>ignore_hidden</p><div aria-hidden="true">ignore_aria</div>
<p style="display: none">ignore_css</p><footer>ignore_footer</footer></body></html>'''
        blocks, warnings = load_document(markup, ".html")
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[-1].text, "Submit in 15 days & keep receipts.")
        self.assertEqual(blocks[-1].locator["anchor"], "deadline")
        self.assertEqual(blocks[-1].locator["section"], ["Policy", "Deadline"])
        self.assertNotIn("ignore", " ".join(block.text.lower() for block in blocks))
        self.assertEqual(warnings, [])

    def test_plain_container_text_and_nested_lists_are_not_lost(self):
        markup = b'<main>Before<div>Middle <strong>value</strong></div><ul><li>Outer<ul><li>Inner</li></ul></li></ul>After</main>'
        blocks, _ = load_document(markup, ".htm")
        text = " ".join(block.text for block in blocks)
        for fragment in ["Before", "Middle value", "Outer", "Inner", "After"]:
            self.assertIn(fragment, text)
        self.assertEqual(text.count("Inner"), 1)
        self.assertEqual(len({block.locator["block"] for block in blocks}), len(blocks))

    def test_invalid_encoding_is_reported(self):
        with self.assertRaises(DocumentReadError):
            load_document(b"\xff\xfe", ".html")


class XLSXTests(unittest.TestCase):
    def test_sheet_cells_zero_false_dates_and_formulas_are_retained(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Limits"
        sheet.append(["Category", "Value", "Note"])
        sheet.append(["Internet", 0, False])
        sheet.append(["Effective date", date(2026, 1, 1), None])
        sheet.append(["Calculated amount", "=SUM(1,2)", None])
        hidden = workbook.create_sheet("Hidden")
        hidden["A1"] = "not for this demo"
        hidden.sheet_state = "hidden"
        blocks, warnings = load_document(xlsx_bytes(workbook), ".xlsx")
        self.assertEqual(len(blocks), 4)
        self.assertIn("B2: 0", blocks[1].text)
        self.assertIn("C2: False", blocks[1].text)
        self.assertIn("2026-01-01", blocks[2].text)
        self.assertIn("[formula, not evaluated] =SUM(1,2)", blocks[3].text)
        self.assertEqual(blocks[1].locator["cell_range"], "A2:C2")
        self.assertEqual(blocks[1].locator["context_range"], "A1:C1")
        self.assertTrue(any("Hidden" in warning for warning in warnings))
        self.assertTrue(any("formula" in warning for warning in warnings))

    def test_blank_rows_do_not_renumber_original_cell_ranges(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet["B3"] = "Limit"
        sheet["B7"] = 1500
        blocks, _ = load_document(xlsx_bytes(workbook), ".xlsx")
        self.assertEqual([block.locator["row"] for block in blocks], [3, 7])
        self.assertEqual(blocks[1].locator["cell_range"], "B7:B7")
        self.assertEqual(blocks[1].locator["context_range"], "B3:B3")

    def test_broken_xlsx_is_an_actionable_document_error(self):
        with self.assertRaises(DocumentReadError):
            load_document(b"not an xlsx", ".xlsx")


if __name__ == "__main__":
    unittest.main()
