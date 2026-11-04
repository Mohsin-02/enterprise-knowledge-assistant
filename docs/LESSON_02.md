# Lesson 2 — parse different file formats and keep evidence locations

Before retrieval, every input must become text plus a location that a person can inspect.
A PDF has pages. A Word file has paragraphs and tables. An Excel file has sheets and cells.
This milestone implements that conversion without an embedding model or LLM.

## The common representation

Each parser returns `ParsedBlock(number, text, locator)`:

- `number`: an actual PDF page number, or None for other formats.
- `text`: cleaned evidence from that part of the source.
- `locator`: source-specific information retained in the saved chunk.

| Source | Example locator |
| --- | --- |
| PDF | page 2 |
| DOCX paragraph | body paragraph 4, under section "Laptop access" |
| DOCX table | table 1, row 2, with first-row context |
| HTML | block 4, anchor `vpn-access`, under section "VPN access" |
| XLSX | sheet `Limits`, data range `A2:C2`, first-row context `A1:C1` |

The `page` field stays null for Word, HTML, and Excel. We do not invent page numbers for
formats whose rendered pagination is unavailable to the parser.

## Follow the code

1. `loaders.py` chooses a parser from the extension and returns common blocks.
2. `office_loaders.py` reads Word body content in order and reads visible Excel sheets row by row.
3. `html_loader.py` uses HTMLParser callbacks to preserve visible blocks, headings, and anchor IDs.
4. `pipeline.py` splits each block into overlapping word windows and copies its locator into each chunk.
5. `inspect_chunks.py` shows the result without requiring you to read the entire JSONL file.

`block_index` is included in the chunk identity. Without it, two identical short paragraphs in
one Word document could accidentally receive the same ID because both start at character zero.
The parser version is recorded so future changes to extraction can be traced.

## Run the lesson

Use the Step 2 project in a fresh folder. Set up its Python environment and install the updated
requirements as shown in README.md. Then run:

```powershell
.\.venv\Scripts\python.exe create_demo_files.py
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe inspect_chunks.py --format xlsx --limit 2
```

The demo generator adds three fictional files and their registry entries. It leaves a complete
existing demo set unchanged on repeat runs and refuses to overwrite a partial/conflicting set.
Without demo generation, the original five documents still ingest; after generation, the expected
result is eight documents and 28 chunks with the default settings.

Inspect the second XLSX chunk. It should contain the monthly internet limit of 1500 in cell B2,
the row locator A2:C2, and first-row context A1:C1. The text identifies both the value and its
table context. This is prepared evidence, not a generated answer.

Repeat inspection for the other new formats:

```powershell
.\.venv\Scripts\python.exe inspect_chunks.py --format docx --limit 3
.\.venv\Scripts\python.exe inspect_chunks.py --format html --limit 4
```

## Scope of the parsers

The current chunker uses source blocks and word windows. Semantic boundary selection using
embeddings will be built and compared with this baseline in Step 3.

DOCX parsing covers body paragraphs and top-level table rows. It repeats the literal first
nonempty table row as context; this is not a claim that every table has formal column headers.
Nested tables and tracked revisions produce warnings. Headers, footers, text boxes, image text,
and detailed layout are outside this parser's current scope.

HTML parsing reads local UTF-8 files. It removes head/script/style/nav/footer/template content,
hidden attributes, aria-hidden elements, and simple inline hiding rules. It does not fetch URLs,
render JavaScript, or resolve external CSS. HTMLParser is not a browser layout engine.

XLSX parsing emits nonempty rows from visible sheets and repeats literal first-row context.
Hidden-sheet skipping is a parsing policy, not an access-control mechanism. Merged cells and
complex tables need inspection; merged text is not propagated across a region. Formulas are
stored as explicitly labeled formula text, because openpyxl does not calculate them. The starter
limits sheets to 50,000 rows and 200 columns. Legacy DOC and XLS require conversion first.

Source offsets refer to the cleaned text block, which can include repeated table context.
Use the locator to trace that context to its original row or cell range.

## Upgrade note

Step 2 uses manifest schema 3 and parser version 2. Chunk IDs may change compared with Step 1
because source-block identity is now included. Rebuild the ingestion outputs with this version.
No vector index was created in the earlier steps.

Access labels are still metadata only. The user-facing retrieval and authentication checks are
later milestones. Actual local execution was checked on Python 3.12; Python 3.14 is your chosen
target and still needs verification on your machine.

## Official references

- Word content order: https://python-docx.readthedocs.io/en/latest/api/document.html
- HTML parsing: https://docs.python.org/3.14/library/html.parser.html
- Excel read-only mode: https://openpyxl.readthedocs.io/en/stable/optimized.html
- Formula behavior: https://openpyxl.readthedocs.io/en/stable/simple_formulae.html
