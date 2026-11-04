"""Dispatch document parsers and preserve the location of each text block."""

from dataclasses import dataclass
from io import BytesIO
import re
import unicodedata

from pypdf import PdfReader
from pypdf.errors import PdfReadError


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".xlsx"}


class DocumentReadError(ValueError):
    """The file could not be converted into reliable text."""


@dataclass(frozen=True)
class ParsedBlock:
    number: int | None  # PDF page numbers start at 1; text files use None.
    text: str
    locator: dict | None = None


def clean_text(text: str) -> str:
    """Normalize encoding and whitespace without deleting paragraph boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(data: bytes, suffix: str) -> tuple[list[ParsedBlock], list[str]]:
    """Return nonempty source blocks and warnings, using the file's format."""
    suffix = suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            text = clean_text(data.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise DocumentReadError("Save this text file as UTF-8 and try again.") from exc
        return ([ParsedBlock(None, text, {"format": suffix[1:], "kind": "text"})] if text else []), []

    if suffix == ".docx":
        from .office_loaders import load_docx
        return load_docx(data)
    if suffix == ".xlsx":
        from .office_loaders import load_xlsx
        return load_xlsx(data)
    if suffix in {".html", ".htm"}:
        from .html_loader import load_html
        return load_html(data)

    if suffix != ".pdf":
        raise DocumentReadError(f"Unsupported file extension: {suffix}")

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentReadError("Password-protected PDF: provide an unlocked copy.")
        pages: list[ParsedBlock] = []
        warnings: list[str] = []
        for number, pdf_page in enumerate(reader.pages, start=1):
            text = clean_text(pdf_page.extract_text() or "")
            if text:
                pages.append(ParsedBlock(number, text, {"format": "pdf", "kind": "page", "page": number}))
            else:
                warnings.append(
                    f"Page {number} has no extractable text; a scanned page may need OCR."
                )
        return pages, warnings
    except DocumentReadError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, IndexError, NotImplementedError) as exc:
        raise DocumentReadError(f"PDF extraction failed ({type(exc).__name__}).") from exc


# Preserve the Step 1 Python imports while the pipeline uses the clearer name.
Page = ParsedBlock
load_pages = load_document
