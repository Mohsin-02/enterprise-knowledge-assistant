"""Add three fictional documents that exercise permissions, Markdown and PDF page locations.

* hr/disciplinary_procedure.txt      - restricted to the hr group
* finance/vendor_payment_controls.md - restricted to the finance group
* security/information_security_handbook.pdf - two pages (needs `pip install reportlab`, dev only)

Existing files and registry entries are never overwritten.
"""

import json
from pathlib import Path

from enterprise_rag.metadata import load_registry
from enterprise_rag.pipeline import atomic_write

DISCIPLINARY = """NOVA DEMO COMPANY - DISCIPLINARY PROCEDURE
Synthetic training document. These are fictional rules for this project. Restricted to Human Resources.

Reporting a concern
A manager who suspects misconduct must submit Form HR-17 to Human Resources within two working days. The form records the dates, the people involved, and any evidence already collected. Managers must not investigate on their own or discuss the concern with the wider team.

Investigation steps
Human Resources assigns an investigator who is not in the employee's reporting line. The investigator interviews the employee within ten working days and records a written summary. The employee may bring a colleague to the interview. Outcomes are limited to no action, a written warning, a final warning, or termination approved by the Chief People Officer.

Appeals
An employee may appeal a written warning or termination to the Appeals Panel within seven calendar days of the outcome letter. The panel decision is final.
"""

VENDOR_CONTROLS = """# Vendor payment controls

Synthetic training document. These are fictional rules for this project. Restricted to Finance.

## Approval limits

Control VPC-4 requires two approvers for any vendor payment above INR 250,000. Payments up to INR 250,000 need one approver from the budget owner's team. The approver and the payment requester must never be the same person.

## Bank detail changes

A change to a vendor's bank account must be confirmed by a call-back to a phone number already on file. Finance must not use contact details supplied in the change request itself. Changes are applied only after the call-back is recorded in the vendor ledger.

## Payment runs

Scheduled payment runs occur every Tuesday and Friday. Urgent payments outside a run need the Finance Controller's written approval.
"""

SECURITY_PAGES = [
    ["Information Security Handbook", "Nova Demo Company. Synthetic training document. These are fictional rules.",
     "Standard SEC-204 requires every employee password to contain at least fourteen characters.",
     "Passwords must be changed immediately if the IT service desk reports a suspected compromise.",
     "Multi-factor authentication is mandatory for the VPN and for the employee portal.",
     "Report a lost badge to Facilities and a lost laptop to the IT service desk without delay."],
    ["Incident handling", "A suspected security incident must be reported to the security hotline on extension 5500.",
     "Do not forward suspicious emails; use the report-phishing button in the mail client instead.",
     "Removable storage devices may be used only after IT has scanned them for malware.",
     "The security team reviews every reported incident within one business day."],
]


def _entries() -> dict:
    def meta(title, family, groups):
        return {"title": title, "document_family": family, "version": "2026", "effective_from": "2026-01-01",
                "effective_to": None, "access_groups": groups}
    return {
        "hr/disciplinary_procedure.txt": meta("Disciplinary procedure", "disciplinary-procedure", ["hr"]),
        "finance/vendor_payment_controls.md": meta("Vendor payment controls", "vendor-payment-controls", ["finance"]),
        "security/information_security_handbook.pdf": meta("Information security handbook",
                                                            "information-security-handbook", ["employees", "it"]),
    }


def _pdf_bytes() -> bytes:
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for lines in SECURITY_PAGES:
        y = 780
        for index, line in enumerate(lines):
            pdf.setFont("Helvetica-Bold" if index == 0 else "Helvetica", 14 if index == 0 else 11)
            pdf.drawString(60, y, line)
            y -= 28
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def create(project: Path) -> bool:
    registry_path = project / "config/documents.json"
    load_registry(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    entries = _entries()
    paths = {source: project / "data/raw" / source for source in entries}
    if all(p.exists() for p in paths.values()) and all(registry["documents"].get(s) == e for s, e in entries.items()):
        return False
    if any(p.exists() for p in paths.values()) or any(s in registry["documents"] for s in entries):
        raise ValueError("Some extra-corpus paths or entries already exist; nothing was changed.")
    contents = {
        "hr/disciplinary_procedure.txt": DISCIPLINARY.encode("utf-8"),
        "finance/vendor_payment_controls.md": VENDOR_CONTROLS.encode("utf-8"),
        "security/information_security_handbook.pdf": _pdf_bytes(),
    }
    for source, data in contents.items():
        paths[source].parent.mkdir(parents=True, exist_ok=True)
        atomic_write(paths[source], data)
    registry["documents"].update(entries)
    atomic_write(registry_path, (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return True


if __name__ == "__main__":
    try:
        print("Created 3 extra documents." if create(Path(__file__).resolve().parent) else "Already present.")
    except (ValueError, OSError, ImportError) as exc:
        raise SystemExit(f"Extra corpus setup stopped: {exc}")
