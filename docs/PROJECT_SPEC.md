# Enterprise Knowledge Intelligence System

Source: the user's project specification supplied on 2026-10-21. This document records the
target requirements. It does not claim that every component is already implemented.

## Architecture and purpose

Intermediate–advanced project combining Advanced RAG, Hybrid RAG, and Conversational RAG.
Search an organization's PDFs, Word documents, Excel files, technical documentation,
knowledge bases, and internal policies, and generate evidence-backed answers.

Target example: with a corpus eventually reaching 5,000 documents, answer:
"What changes were made to the employee reimbursement policy between 2024 and 2026?"
The result must identify the relevant revisions, compare their contents, and cite supporting
passages from each. Step 2 has eight fictional documents after demo generation; scale has not been benchmarked.

## Required processing stages

Document ingestion and format parsing; text cleaning; semantic chunking; embeddings; parallel
Qdrant vector retrieval and BM25 keyword retrieval; reciprocal-rank fusion; cross-encoder
reranking; top-K context selection; grounded LLM answers; source citations.

Conversational follow-up questions must be interpreted using session history and answered
with newly retrieved evidence. Historical policy comparisons must retain separate revisions.
Retrieval must respect the authenticated employee's document permissions.

## Target technology stack

| Component | Technology |
| --- | --- |
| Language | Python |
| API | FastAPI |
| Orchestration | LangChain |
| Embeddings | BGE-M3 or a suitable smaller baseline |
| Vector storage | Qdrant |
| Keyword retrieval | BM25 |
| Ranking fusion | Reciprocal Rank Fusion |
| Reranking | Cross-encoder |
| Answer generation | Qwen initially; Llama or GPT are alternatives in the supplied specification |
| Metadata and sessions | PostgreSQL |
| Initial interface | Streamlit; React remains an optional later alternative |
| Packaging and deployment | Docker |

The open-model path selected for this tutorial does not require an OpenAI API key.
We will select exact model versions and resource settings when implementing those stages.

## Engineering requirements

- Semantic chunking must use meaningful content boundaries. Keep the current word-window
  approach as a baseline, and compare chunking strategies with labeled retrieval questions.
- Hybrid retrieval must combine lexical matching and semantic similarity. Apply the same
  permission and version constraints to both candidate sets before combining rankings.
- RRF must combine rank positions explicitly; raw BM25 and vector scores are not assumed
  to share a comparable scale.
- Reranking must score query-passage pairs and record quality and latency tradeoffs.
- Each factual answer claim must have supporting source passages. Validate that a cited
  identifier refers to retrieved evidence; unsupported questions require appropriate abstention.
- Version comparisons need a shared document-family identifier, revision labels, effective
  dates, and evidence from the requested revisions. Do not substitute the latest policy for
  an older requested revision or claim there was no change when one side is missing.
- The server must authenticate users and derive permissions from trusted records. Keep
  unauthorized passages out of candidate results, model prompts, citations, caches, and history.
- Session memory must preserve follow-up context without carrying documents between users.
- Ingestion must support stable identities, version retention, metadata updates, deletion
  handling, and repeatable runs as the application matures.

## Evaluation requirements

Measure Recall@K, Precision@K, MRR, NDCG, answer faithfulness, citation accuracy, and retrieval
latency. Use documented relevance labels and a held-out question set. Include ordinary lookup,
exact identifiers, paraphrases, cross-document questions, version comparisons, conversational
follow-ups, unsupported questions, and permission-denied cases.

Compare dense-only, BM25-only, hybrid, and hybrid-with-reranking configurations. Record model
versions, chunk settings, candidate counts, and serving hardware alongside results. Use human
inspection for a subset of answer and citation checks; an LLM judge is supplementary evidence.

## Representative completion checks

1. A normal leave question retrieves the relevant leave policy and cites its source.
2. The reimbursement comparison cites both the 2024 and 2026 revisions.
3. A follow-up such as "What about the submission deadline?" retains the intended topic and
   requested revisions, then retrieves evidence for that narrower question.
4. A user without an authorized group receives no restricted passage or document-existence leak.
5. A question with no supporting source does not receive a fabricated policy answer.
6. A metadata-only permission update refreshes retrieval payloads and invalidates affected caches.
7. Every supported format has usable source locations: PDF pages, DOCX paragraph/section
   locations, HTML headings/anchors, and Excel sheet/cell ranges.

