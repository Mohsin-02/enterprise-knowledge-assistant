# Enterprise Knowledge Intelligence System — learning roadmap

Follow the full user specification in `PROJECT_SPEC.md`. Complete, explain, and verify one
milestone at a time. The current download completes the parsing milestone in Step 2.

## Target behavior

An employee asks what changed in the reimbursement policy between 2024 and 2026. The finished
application retrieves passages from both authorized revisions, compares their rules, and cites
each factual comparison. A follow-up such as "What about the submission deadline?" keeps the
intended context and performs a new retrieval. Unsupported questions receive an appropriate
insufficient-evidence response.

## Milestones

| Step | Deliverable | Completion check |
| --- | --- | --- |
| 1 — foundation built | PDF/text loading, word-window baseline, source versions/dates, access labels | Five demo documents produce eight readable chunks, with both reimbursement revisions retained. |
| 2 — built | DOCX, HTML, and XLSX parsing with source locations | Eight sample documents produce 28 chunks after demo generation; source references are inspectable. |
| 3 | Token-aware semantic chunking, open embeddings, and Qdrant | Compare semantic boundaries with the baseline; queries retrieve relevant passages. |
| 4 | BM25, RRF hybrid retrieval, and permission/version filters | Both retrieval branches use the same constraints; unauthorized passages never reach fusion or prompts. |
| 5 | Cross-encoder reranking | Compare relevance and latency against hybrid search without reranking. |
| 6 | LangChain orchestration, Qwen answers, citations, and version-comparison routing | Compare the 2024/2026 policies with evidence from both; handle missing evidence. |
| 7 | Conversational RAG and PostgreSQL metadata/session persistence | Follow-ups retain topic and requested dates; sessions remain isolated and permissions are checked again. |
| 8 | Evaluation and regression cases | Report Recall@K, Precision@K, MRR, NDCG, faithfulness, citation accuracy, and serving latency. |
| 9 | FastAPI, authenticated identity, Streamlit, and Docker deployment | Run the complete upload/question/evidence flow with server-derived groups, persistent storage, and permission tests. |

## Computing approach

Run ingestion and the small application on the laptop. Use a small open embedding model for
an initial CPU baseline; compare a larger model in Colab when hardware is available.
For generation, start with a suitably sized Qwen open instruction model. The exact model and
quantization settings will be selected at that milestone after checking hardware and current
compatibility. No OpenAI integration is included in Step 1.

Colab is useful for interactive embedding, extraction, and evaluation experiments. GPU type,
availability, and session limits are variable. Save results outside the temporary runtime and
make longer work resumable. Use a persistent environment for the eventual deployed app.

Start with five fictional documents, then a small real corpus. A large document count is not a
substitute for labeled retrieval questions and inspected evidence. Use the same model version,
pooling, normalization, and query/document encoding rules when building and querying an index.

## Scope notes

This starter parses `.txt`, `.md`, text-based PDFs, DOCX, local HTML, and XLSX. It does not yet implement semantic
chunking, embeddings, search, answer generation, version-comparison
logic, conversational memory, permission enforcement, authentication, PostgreSQL, or a frontend.
Access labels and effective dates are stored and validated; the runtime behavior using them
will be built in later milestones. There is no model training or model download in this step.
The code has been checked in the assistant's workspace and has not been run on your laptop.

An LLM judge can supplement evaluation later; it is not ground truth. Inspect evidence and
manually label a small question set as well. Measure serving latency on the environment used
for the demo, and label separate Colab experiment timings by their actual hardware.
