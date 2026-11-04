# Step 2 verification

Verified on 2026-10-22 in the assistant's Linux workspace with Python 3.12.14, pypdf 6.10.0,
python-docx 1.2.0, openpyxl 3.1.5, and lxml 6.1.1.

- All twenty-four unit tests passed.
- The eight unmodified sample documents produced 28 chunks after demo generation, with zero failed, empty, or
  unregistered documents.
- Chunk IDs were unique; rerunning unchanged input produced identical outputs.
- A generated PDF with a blank first page and text on pages two and three preserved the original
  page numbers and reported the blank page.
- The saved chunk file matched the SHA-256 digest in its manifest.
- Invalid overlap settings left previous outputs unchanged and returned a failing exit code.
- A partial run with one malformed PDF returned exit code 2 and recorded the failure.
- Both reimbursement revisions retained their document-family, version, date, and access-group fields.
- Identical passage text in different policy revisions was retained as separate evidence.
- Updating only access labels preserved text identity and updated the saved metadata digest.
- Invalid dates and access-group types were rejected before replacing previous outputs.
- A source missing from the configured registry was excluded and recorded as a failure.
- DOCX paragraph/table order and paragraph/table-row locations were preserved.
- Identical DOCX paragraphs received distinct stable chunk IDs through their block locations.
- HTML heading anchors and nested visible content were retained, while selected noncontent was removed.
- XLSX sheet/cell references, zero, False, and dates were retained; formulas were explicitly labeled.
- Blank Excel rows did not renumber the original source locations.

Windows commands are supplied, but this run used Linux. The starter has not been executed on
the user's laptop or on Python 3.14 here. No semantic chunking, embedding model, vector search, answer-generation,
conversational memory, or permission-enforcement stage is included yet.
