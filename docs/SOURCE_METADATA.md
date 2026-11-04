# Lesson 1 — source versions and access labels

The original project example asks what changed between reimbursement policies from 2024
and 2026. Searching text alone cannot reliably tell which revision a passage belongs to.
That is why each source is registered before the command-line ingestion run.

## A registration entry

In `config/documents.json`, the keys inside `documents` match paths relative to `data/raw/`.
For example, `hr/policy.pdf` identifies `data/raw/hr/policy.pdf`.

```json
"reimbursement_policy_2026.txt": {
  "title": "Employee reimbursement policy",
  "document_family": "reimbursement-policy",
  "version": "2026",
  "effective_from": "2026-01-01",
  "effective_to": null,
  "access_groups": ["employees", "finance", "hr"]
}
```

This is an entry inside the existing `documents` object, not an entire standalone JSON file.

| Field | Meaning |
| --- | --- |
| `title` | A readable name for the document. |
| `document_family` | The shared identity of all revisions of the same policy. |
| `version` | A revision label, stored as a string. It does not need to be a year. |
| `effective_from` | First date this revision is effective, in YYYY-MM-DD format. |
| `effective_to` | Last effective date, inclusive; null means no end date is recorded. |
| `access_groups` | Groups allowed by the source metadata; an empty list grants no groups. |

These dates must come from source records. A version label is not evidence of an effective date.
Do not fabricate missing dates for real documents. This tutorial's dates and group names are
explicitly fictional. The first registry requires known start dates; we can extend the schema
for unknown dates when a real corpus requires it.

## What happens during ingestion

1. Validate the registry and dates before writing outputs.
2. Match each input file to its relative source path.
3. Attach source metadata to every chunk from that file.
4. Save text identity and metadata identity separately.

The CLI rejects an unregistered source rather than assigning public access. Calling the Python
API directly without a registry is allowed for experiments, but produces unregistered chunks
with no access groups. Later serving code must exclude those chunks.

Changing an access label does not change the passage text, so `chunk_id` stays stable while
`metadata_sha256` changes. This will let a future index reuse embeddings and update metadata
payloads. Such payload synchronization and cache invalidation are not implemented in this step.

## Inspect the two sample revisions

After running `ingest.py`, find the two reimbursement sources in `data/processed/chunks.jsonl`.
They should share `document_family` and have different `version` and date values.

The fictional source documents contain these changes:

| Rule | 2024 revision | 2026 revision |
| --- | --- | --- |
| Annual reimbursement cap | INR 30,000 | INR 45,000 |
| Monthly internet limit | INR 1,000 | INR 1,500 |
| Claim submission window | 30 calendar days | 15 calendar days |
| Invoice evidence | Original paper invoices | Uploaded digital invoices accepted |
| Prior Finance approval | No additional threshold stated | Required for an individual expense exceeding INR 10,000 |

This table is a manually inspected expectation, not a generated answer or measured model result.
Later retrieval and generation should locate these passages and cite both revisions.

## Limits of this lesson

Storing groups is not access control. The later application must authenticate the employee,
derive groups on the server, filter both dense and keyword retrieval before fusion or reranking,
and prevent unauthorized content from entering prompts, answers, caches, or conversation history.
The current local ingestion files are developer artifacts and are not a user-facing access gate.
