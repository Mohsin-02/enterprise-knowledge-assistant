"""Parse DOCX body content and flat XLSX sheets with useful source locations."""

from datetime import date, datetime, time
from io import BytesIO
import re
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import InvalidFileException

from .loaders import DocumentReadError, ParsedBlock, clean_text


def load_docx(data: bytes) -> tuple[list[ParsedBlock], list[str]]:
    """Keep top-level paragraphs and table rows in their original body order."""
    try:
        document = Document(BytesIO(data))
        blocks: list[ParsedBlock] = []
        warnings: list[str] = []
        headings: dict[int, str] = {}
        paragraph_number = table_number = 0
        for body_index, item in enumerate(document.iter_inner_content(), start=1):
            if isinstance(item, Paragraph):
                paragraph_number += 1
                text = clean_text(item.text)
                if not text:
                    continue
                heading = re.fullmatch(r"Heading ([1-9])", item.style.name or "")
                if heading:
                    level = int(heading.group(1))
                    headings = {key: value for key, value in headings.items() if key < level}
                    headings[level] = text
                blocks.append(ParsedBlock(None, text, {
                    "format": "docx", "kind": "paragraph", "body_block": body_index,
                    "paragraph": paragraph_number, "section": list(headings.values()),
                }))
            else:
                table_number += 1
                first_row = None
                for row_number, row in enumerate(item.rows, start=1):
                    cells: list[str] = []
                    visited: set[int] = set()
                    for cell in row.cells:
                        cell_id = id(cell._tc)
                        if cell_id in visited:
                            continue
                        visited.add(cell_id)
                        cells.append(clean_text(cell.text))
                        if cell.tables:
                            warnings.append(
                                f"Table {table_number}, row {row_number}: nested table content is not parsed."
                            )
                    if not any(cells):
                        continue
                    row_text = " | ".join(cells)
                    if first_row is None:
                        first_row = (row_number, row_text)
                    # Repeat the literal first row as context; do not invent column labels.
                    text = row_text if row_number == first_row[0] else (
                        f"First-row context: {first_row[1]}\nRow {row_number}: {row_text}"
                    )
                    blocks.append(ParsedBlock(None, text, {
                        "format": "docx", "kind": "table_row", "body_block": body_index,
                        "table": table_number, "row": row_number,
                        "context_row": first_row[0], "section": list(headings.values()),
                    }))
        if document.element.body.xpath(".//w:ins | .//w:del"):
            warnings.append("Tracked revisions detected; accept or reject changes in Word before indexing.")
        return blocks, warnings
    except (BadZipFile, PackageNotFoundError, XMLSyntaxError, KeyError, ValueError, EOFError) as exc:
        raise DocumentReadError(f"DOCX extraction failed ({type(exc).__name__}).") from exc


def _cell_text(cell) -> str:
    value = cell.value
    if cell.data_type == "f":
        return f"[formula, not evaluated] {value}"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return clean_text(str(value))


def load_xlsx(data: bytes) -> tuple[list[ParsedBlock], list[str]]:
    """Emit one block per nonempty row, preserving cells, zeros and formulas."""
    workbook = None
    try:
        workbook = load_workbook(BytesIO(data), read_only=True, data_only=False, keep_links=False)
        blocks: list[ParsedBlock] = []
        warnings: list[str] = []
        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                warnings.append(f"Hidden sheet {sheet.title!r} was skipped.")
                continue
            if (sheet.max_row or 0) > 50000 or (sheet.max_column or 0) > 200:
                raise DocumentReadError(f"Sheet {sheet.title!r} exceeds the starter's 50,000-row/200-column limit.")
            first_row = None
            formula_example = None
            for row_number, row in enumerate(sheet.iter_rows(), start=1):
                cells = [cell for cell in row if cell.value is not None and str(cell.value).strip()]
                if not cells:
                    continue
                start = f"{get_column_letter(cells[0].column)}{row_number}"
                end = f"{get_column_letter(cells[-1].column)}{row_number}"
                row_range = f"{start}:{end}"
                row_text = " | ".join(f"{cell.coordinate}: {_cell_text(cell)}" for cell in cells)
                if first_row is None:
                    first_row = (row_range, row_text)
                if formula_example is None:
                    formula_example = next((cell.coordinate for cell in cells if cell.data_type == "f"), None)
                text = f"Sheet: {sheet.title}\n"
                if row_range != first_row[0]:
                    text += f"First-row context ({first_row[0]}): {first_row[1]}\n"
                text += row_text
                blocks.append(ParsedBlock(None, text, {
                    "format": "xlsx", "kind": "sheet_row", "sheet": sheet.title,
                    "row": row_number, "cell_range": row_range, "context_range": first_row[0],
                }))
            if formula_example is not None:
                warnings.append(
                    f"Sheet {sheet.title!r} contains formulas (e.g. {formula_example}); "
                    "they are stored as formula text, not calculated values."
                )
        return blocks, warnings
    except DocumentReadError:
        raise
    except (BadZipFile, InvalidFileException, XMLSyntaxError, KeyError, ValueError, EOFError, TypeError) as exc:
        raise DocumentReadError(f"XLSX extraction failed ({type(exc).__name__}).") from exc
    finally:
        if workbook is not None:
            workbook.close()
