"""Create three fictional parser fixtures without overwriting existing files."""

from io import BytesIO
import json
from pathlib import Path

from docx import Document
from openpyxl import Workbook

from enterprise_rag.metadata import load_registry
from enterprise_rag.pipeline import atomic_write


HTML_DEMO = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Demo knowledge base</title></head>
<body><nav>Navigation boilerplate</nav><main>
<h1 id="it-help">IT knowledge base</h1>
<p>Synthetic training document. These company rules are fictional.</p>
<h2 id="vpn-access">VPN access</h2>
<p>Employees working remotely must connect to the approved VPN before opening internal tools.
Request VPN access through the IT service portal and obtain reporting-manager approval.</p>
<h2 id="password-reset">Password reset</h2>
<p>Use the self-service password reset page. If the account is locked, contact the IT service desk
with the original support ticket number. Never send your password in a ticket.</p>
</main><script>this text must not enter the corpus</script><footer>Footer boilerplate</footer></body></html>
"""


def _metadata(title: str, family: str) -> dict:
    return {"title": title, "document_family": family, "version": "2026",
            "effective_from": "2026-01-01", "effective_to": None,
            "access_groups": ["employees"]}


def create_demo_files(project: Path) -> bool:
    registry_path = project / "config/documents.json"
    load_registry(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    entries = {
        "demo/onboarding.docx": _metadata("Employee onboarding", "onboarding-guide"),
        "demo/knowledge_base.html": _metadata("IT knowledge base", "it-knowledge-base"),
        "demo/expense_limits.xlsx": _metadata("Reimbursement limit table", "expense-limit-table"),
    }
    paths = [project / "data/raw" / source for source in entries]
    registered = [registry["documents"].get(source) == entry for source, entry in entries.items()]
    if all(path.exists() for path in paths) and all(registered):
        return False
    if any(path.exists() for path in paths) or any(source in registry["documents"] for source in entries):
        raise ValueError("Some demo paths or entries already exist. Use a fresh project folder; existing files were kept.")

    document = Document()
    document.add_heading("Employee onboarding", level=1)
    document.add_paragraph("Synthetic training document. These company rules are fictional.")
    document.add_heading("Laptop access", level=2)
    document.add_paragraph("New employees request a managed laptop through IT within their first three working days.")
    document.add_heading("Approval", level=2)
    document.add_paragraph("The reporting manager approves the equipment request in the employee portal.")
    table = document.add_table(rows=3, cols=2)
    for row, values in zip(table.rows, [("Item", "Owner"), ("Laptop", "IT service desk"), ("Office badge", "Facilities")]):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.add_paragraph("Contact Human Resources if the onboarding checklist is incomplete after the first week.")
    docx_bytes = BytesIO()
    document.save(docx_bytes)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Limits"
    for row in [("Category", "Limit (INR)", "Approval"),
                ("Monthly internet", 1500, "Reporting manager"),
                ("Annual reimbursement cap", 45000, "Finance"),
                ("Individual expense threshold", 10000, "Prior Finance approval above this amount")]:
        sheet.append(row)
    xlsx_bytes = BytesIO()
    workbook.save(xlsx_bytes)
    workbook.close()
    contents = [docx_bytes.getvalue(), HTML_DEMO.encode("utf-8"), xlsx_bytes.getvalue()]
    for path, data in zip(paths, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, data)
    registry["documents"].update(entries)
    atomic_write(registry_path, (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return True


if __name__ == "__main__":
    try:
        created = create_demo_files(Path(__file__).resolve().parent)
        print("Created 3 demo files and registered their metadata." if created else "Demo files already exist; kept them unchanged.")
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Demo setup stopped: {exc}")
